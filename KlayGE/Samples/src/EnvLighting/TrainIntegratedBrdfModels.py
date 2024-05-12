import argparse
import os
import math
import struct
import time

import torch
import torch.nn as nn
import torch.optim as optim

class IntegratedBrdfDataset:
	def __init__(self, device, file_name):
		with open(file_name, "rb") as file:
			sample_count = struct.unpack("<I", file.read(4))[0]
			self.coord_data = torch.empty(sample_count, 2)
			self.x_data = torch.empty(sample_count)
			self.y_data = torch.empty(sample_count)
			sample_format = "<ffff"
			buffer = file.read(sample_count * struct.calcsize(sample_format))
			unpack_iter = struct.iter_unpack(sample_format, buffer)
			for i, sample in enumerate(unpack_iter):
				self.coord_data[i] = torch.tensor(sample[0:2])
				self.x_data[i] = sample[2]
				self.y_data[i] = sample[3]

			self.coord_data = self.coord_data.to(device)
			self.x_data = self.x_data.to(device)
			self.y_data = self.y_data.to(device)

			self.output_data = self.x_data

	def XChannelMode(self, x_mode):
		self.output_data = self.x_data if x_mode else self.y_data

	def __len__(self):
		return len(self.coord_data)

class IntegratedBrdfMlpNetwork(nn.Module):
	def __init__(self, num_features):
		super(IntegratedBrdfMlpNetwork, self).__init__()

		layers = []
		for i in range(len(num_features) - 1):
			layers.append(nn.Linear(num_features[i], num_features[i + 1]))
			if i != len(num_features) - 2:
				layers.append(nn.Sigmoid())

		self.net = nn.Sequential(*layers)

	def forward(self, x):
		return self.net(x).squeeze(1)

	def Write(self, file, var_name):
		file.write(f"/*{var_name} structure: {self}*/\n\n")

		for i in range((len(self.net) + 1) // 2):
			weight = self.net[i * 2].weight.data
			dim0 = weight.size()[0]
			dim1 = weight.size()[1]
			file.write(f"float const {var_name}_layer_{i + 1}_weight[][{dim1}] = {{\n")
			for row_index, row in enumerate(weight):
				file.write("\t{")
				for col_index, col in enumerate(row):
					file.write(f"{col:.9f}f")
					if ((row_index != dim0 - 1) or (col_index != dim1 - 1)) and (col_index != dim1 - 1):
						file.write(", ")
				file.write("},\n")
			file.write("};\n")

			bias = self.net[i * 2].bias.data
			dim0 = bias.size()[0]
			file.write(f"float const {var_name}_layer_{i + 1}_bias[]{{")
			for col_index, col in enumerate(bias):
				file.write(f"{col:.9f}f")
				if col_index != dim0 - 1:
					file.write(", ")
			file.write("};\n\n")

class IntegratedBrdfExpression(nn.Module):
	def __init__(self, order, init_factors = None):
		super(IntegratedBrdfExpression, self).__init__()

		self.order = order
		n = self.order + 1

		num_factors = n * n
		if init_factors == None:
			init_factors = torch.distributions.Uniform(0, 5).sample((num_factors, ))
		else:
			init_factors = init_factors.t().reshape(num_factors)

		self.weights = nn.Parameter(init_factors)

	def forward(self, x):
		n_dot_v, glossiness = torch.tensor_split(x, 2, dim = 1)
		weights = self.weights
		n = self.order + 1
		dim_x = []
		for y in range(n):
			tmp = weights[y * n]
			for x in range(1, n):
				tmp = torch.addcmul(weights[y * n + x], tmp, glossiness)
			dim_x.append(tmp)
		dim_y = dim_x[0]
		for x in range(1, n):
			dim_y = torch.addcmul(dim_x[x], dim_y, n_dot_v)
		return dim_y.squeeze(1)

	def Write(self, file, var_name):
		n = self.order + 1
		weights = self.weights
		file.write(f"float{n} const {var_name}[] = {{\n")
		for y in range(n):
			file.write("\t{")
			for x in range(n):
				file.write(f"{weights[x * n + y]:.9f}f")
				if (x != n - 1):
					file.write(", ")
			file.write("},\n")
		file.write("};\n")

# Port from https://github.com/Blealtan/efficient-kan/blob/master/src/efficient_kan/kan.py
class KanLinear(torch.nn.Module):
	def __init__(
		self,
		in_features,
		out_features,
		grid_size = 5,
		spline_order = 3,
		scale_noise = 0.1,
		scale_base = 1.0,
		scale_spline = 1.0,
		enable_standalone_scale_spline = True,
		base_activation = torch.nn.SiLU,
		grid_eps = 0.02,
		grid_range = [-1, 1]) :
		assert grid_size > 0

		super(KanLinear, self).__init__()

		self.in_features = in_features
		self.out_features = out_features
		self.grid_size = grid_size
		self.spline_order = spline_order
		self.scale_noise = scale_noise
		self.scale_base = scale_base
		self.scale_spline = scale_spline
		self.enable_standalone_scale_spline = enable_standalone_scale_spline
		self.base_activation = base_activation()
		self.grid_eps = grid_eps

		h = (grid_range[1] - grid_range[0]) / grid_size
		grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]).expand(in_features, -1).contiguous()
		self.register_buffer("grid", grid)

		self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
		self.spline_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
		if self.enable_standalone_scale_spline:
			self.spline_scaler = torch.nn.Parameter(torch.Tensor(out_features, in_features))

		self.ResetParameters()

	def ResetParameters(self):
		torch.nn.init.kaiming_uniform_(self.base_weight, a = math.sqrt(5) * self.scale_base)
		with torch.no_grad():
			noise = (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 1 / 2) * self.scale_noise / self.grid_size
			spline_weight = self.Curve2Coeff(self.grid.T[self.spline_order : -self.spline_order], noise)
			if not self.enable_standalone_scale_spline:
				spline_weight *= self.scale_spline
			self.spline_weight.data.copy_(spline_weight)
			if self.enable_standalone_scale_spline:
				torch.nn.init.kaiming_uniform_(self.spline_scaler, a = math.sqrt(5) * self.scale_spline)

	def BSplines(self, x: torch.Tensor):
		"""
		Compute the B-spline bases for the given input tensor.

		Args:
			x (torch.Tensor): Input tensor of shape (batch_size, in_features).

		Returns:
			torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
		"""
		assert x.dim() == 2 and x.size(1) == self.in_features

		grid: torch.Tensor = self.grid  # (in_features, grid_size + 2 * spline_order + 1)
		x = x.unsqueeze(-1)
		bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
		for k in range(1, self.spline_order + 1):
			bases = (
				(x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]
			) + (
				(grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:-k]) * bases[:, :, 1:]
			)

		assert bases.size() == (
			x.size(0),
			self.in_features,
			self.grid_size + self.spline_order,
		)
		return bases.contiguous()

	def Curve2Coeff(self, x: torch.Tensor, y: torch.Tensor):
		"""
		Compute the coefficients of the curve that interpolates the given points.

		Args:
			x (torch.Tensor): Input tensor of shape (batch_size, in_features).
			y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

		Returns:
			torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
		"""
		assert x.dim() == 2 and x.size(1) == self.in_features
		assert y.size() == (x.size(0), self.in_features, self.out_features)

		A = self.BSplines(x).transpose(0, 1)  # (in_features, batch_size, grid_size + spline_order)
		B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
		solution = torch.linalg.lstsq(A, B).solution  # (in_features, grid_size + spline_order, out_features)
		result = solution.permute(2, 0, 1)  # (out_features, in_features, grid_size + spline_order)

		assert result.size() == (
			self.out_features,
			self.in_features,
			self.grid_size + self.spline_order,
		)
		return result.contiguous()

	@property
	def scaled_spline_weight(self):
		if self.enable_standalone_scale_spline:
			return self.spline_weight * self.spline_scaler.unsqueeze(-1)
		else:
			return self.spline_weight

	def forward(self, x: torch.Tensor):
		assert x.dim() == 2 and x.size(1) == self.in_features

		base_output = nn.functional.linear(self.base_activation(x), self.base_weight)
		spline_output = nn.functional.linear(self.BSplines(x).view(x.size(0), -1), self.scaled_spline_weight.view(self.out_features, -1))
		return base_output + spline_output

class IntegratedBrdfKanNetwork(nn.Module):
	def __init__(self, num_features):
		super(IntegratedBrdfKanNetwork, self).__init__()

		layers = []
		for i in range(len(num_features) - 1):
			layers.append(KanLinear(num_features[i], num_features[i + 1], grid_size = 1, spline_order = 2, base_activation = torch.nn.ReLU))

		self.net = nn.Sequential(*layers)

	def forward(self, x):
		return self.net(x).squeeze(1)

	def Write(self, file, var_name):
		file.write(f"/*{var_name} structure: {self}*/\n\n")

		for layer_index, layer in enumerate(self.net):
			base_weight = layer.base_weight.data
			dim0 = base_weight.size()[0]
			dim1 = base_weight.size()[1]
			file.write(f"float const {var_name}_layer_{layer_index + 1}_base_weight[][{dim1}] = {{\n")
			for row_index, row in enumerate(base_weight):
				file.write("\t{")
				for col_index, col in enumerate(row):
					file.write(f"{col:.9f}f")
					if ((row_index != dim0 - 1) or (col_index != dim1 - 1)) and (col_index != dim1 - 1):
						file.write(", ")
				file.write("},\n")
			file.write("};\n")
		
			scaled_spline_weight = layer.scaled_spline_weight
			dim0 = scaled_spline_weight.size()[0]
			dim1 = scaled_spline_weight.size()[1]
			dim2 = scaled_spline_weight.size()[2]
			file.write(f"float const {var_name}_layer_{layer_index + 1}_scaled_spline_weight[][{dim1}][{dim2}] = {{\n")
			for row_index, row in enumerate(scaled_spline_weight):
				file.write("\t{\n")
				for col_index, col in enumerate(row):
					file.write("\t\t{")
					for item_index, item in enumerate(col):
						file.write(f"{item:.9f}f")
						if ((row_index != dim0 - 1) or (col_index != dim1 - 1) or (item_index != dim2 - 1)) and (item_index != dim2 - 1):
							file.write(", ")
					file.write("},\n")
				file.write("\t},\n")
			file.write("};\n\n")

class ModelDesc:
	def __init__(self, name, model_class, model_param, x_channel_mode, output_file_name):
		self.name = name
		self.model_class = model_class
		self.model_param = model_param
		self.x_channel_mode = x_channel_mode
		self.output_file_name = output_file_name

def TrainModel(device, data_set, model_desc, batch_size, learning_rate, epochs, continue_mode):
	model = model_desc.model_class(model_desc.model_param)

	pth_file_name = model_desc.output_file_name + ".pth"
	if continue_mode and os.path.exists(pth_file_name):
		model.load_state_dict(torch.load(pth_file_name))

	model.train(True)
	model.to(device)

	criterion = nn.MSELoss(reduction = "sum")
	optimizer = optim.Adam(model.parameters(), lr = learning_rate)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor = 0.5, verbose = True)

	start = time.time()
	min_loss = 1e10
	for epoch in range(epochs):
		running_loss = torch.zeros(1, device = device)
		for batch_start in range(0, len(data_set), batch_size):
			inputs = data_set.coord_data[batch_start : batch_start + batch_size]
			targets = data_set.output_data[batch_start : batch_start + batch_size]

			outputs = model(inputs)
			loss = criterion(outputs, targets)
			running_loss += loss

			optimizer.zero_grad(set_to_none = True)
			loss.backward()
			optimizer.step()
		loss = running_loss.item() / len(data_set)
		scheduler.step(loss)
		if loss < min_loss:
			torch.save(model.state_dict(), pth_file_name)
			with open(model_desc.output_file_name + ".hpp", "w") as file:
				model.Write(file, model_desc.name)
				file.write(f"// [{epoch + 1}] Loss: {loss}\n")
			min_loss = loss
		print(f"[{epoch + 1}] Loss: {loss}")
		if loss < 1e-7:
			break
	timespan = time.time() - start

	print(f"Finished training in {(timespan / 60):.2f} mins.")
	print(f"Min loss: {min_loss}")
	print(f"Last learning rate: {optimizer.param_groups[0]['lr']}")

	return model

def TestModel(device, data_set, model, batch_size):
	model.train(False)
	model.to(device)

	total_mse = torch.tensor(0.0, device = device)
	with torch.no_grad():
		for batch_start in range(0, len(data_set), batch_size):
			inputs = data_set.coord_data[batch_start : batch_start + batch_size]
			targets = data_set.output_data[batch_start : batch_start + batch_size]

			outputs = model(inputs)
			diff = outputs - targets
			total_mse += torch.sum(diff * diff)

	return total_mse.item() / len(data_set)

def TrainModels(device, training_set, testing_set, batch_size, learning_rate, epochs, continue_mode, model_descs):
	models = []
	for model_desc in model_descs:
		training_set.XChannelMode(model_desc.x_channel_mode)
		model = TrainModel(device, training_set, model_desc, batch_size, learning_rate, epochs, continue_mode)
		models.append(model)

	for model, model_desc in zip(models, model_descs):
		testing_set.XChannelMode(model_desc.x_channel_mode)
		mse = TestModel(device, testing_set, model, batch_size)
		print(f"MSE of the {model_desc.name} on the {len(testing_set)} test samples: {mse}")

def ParseCommandLine():
	parser = argparse.ArgumentParser()

	parser.add_argument("--batch-size", dest = "batch_size", default = 256, type = int, help = "batch size in training")
	parser.add_argument("--learning-rate", dest = "learning_rate", default = 0.01, type = float, help = "epochs in training")
	parser.add_argument("--epochs", dest = "epochs", default = 500, type = int, help = "epochs in training")
	parser.add_argument("--continue", dest = "continue_mode", default = False, action = "store_true", help = "continue training from current pth file")

	return parser.parse_args()

if __name__ == "__main__":
	args = ParseCommandLine()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	print("Loading training set...")
	training_set = IntegratedBrdfDataset(device, "IntegratedBRDF_1048576.dat")

	idx = torch.randperm(training_set.coord_data.shape[0])
	training_set.coord_data = training_set.coord_data[idx]
	training_set.x_data = training_set.x_data[idx]
	training_set.y_data = training_set.y_data[idx]

	print("Loading testing set...")
	testing_set = IntegratedBrdfDataset(device, "IntegratedBRDF_4096.dat")

	model_descs = (
		(
			"MLP neural network",
			(
				ModelDesc("x_mlp", IntegratedBrdfMlpNetwork, (2, 2, 2, 1), True, "FittedBrdfMlpNN4LayerX"),
				ModelDesc("y_mlp", IntegratedBrdfMlpNetwork, (2, 3, 1), False, "FittedBrdfMlpNN3LayerY"),
			),
		),
		(
			"3-Order expression",
			(
				ModelDesc("x_factors", IntegratedBrdfExpression, 3, True, "FittedBrdfFactorsX"),
				ModelDesc("y_factors", IntegratedBrdfExpression, 3, False, "FittedBrdfFactorsY"),
			),
		),
		(
			"KAN neural network",
			(
				ModelDesc("x_kan", IntegratedBrdfKanNetwork, (2, 2, 1), True, "FittedBrdfKanNN3LayerX"),
				ModelDesc("y_kan", IntegratedBrdfKanNetwork, (2, 2, 1), False, "FittedBrdfKanNN3LayerY"),
			),
		),
	)
	for model in model_descs:
		print(f"Training {model[0]} ...")
		TrainModels(device, training_set, testing_set, args.batch_size, args.learning_rate, args.epochs, args.continue_mode, model[1])

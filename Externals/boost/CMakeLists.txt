UpdateExternalLib("assert" "https://github.com/boostorg/assert.git" "447e0b3a331930f8708ade0e42683d12de9dfbc3")
UpdateExternalLib("core" "https://github.com/boostorg/core.git" "8cc2fdad76b3e2ba67fe6b5dd3c44ee29dd2ab29")

# Depended by assert, core, static_assert, throw_exception
UpdateExternalLib("config" "https://github.com/boostorg/config.git" "11385ec21012926e15a612e3bf9f9a71403c1e5b")
# Depended by core
UpdateExternalLib("static_assert" "https://github.com/boostorg/static_assert.git" "ba72d3340f3dc6e773868107f35902292f84b07e")
# Depended by core
UpdateExternalLib("throw_exception" "https://github.com/boostorg/throw_exception.git" "7c8ec2114bc1f9ab2a8afbd629b96fbdd5901294")

add_subdirectory(assert)
add_subdirectory(core)

add_subdirectory(config)
add_subdirectory(static_assert)
add_subdirectory(throw_exception)

set_target_properties(boost_assert boost_core PROPERTIES
	FOLDER "Externals/boost"
)

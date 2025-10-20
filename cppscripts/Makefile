default: cli

cli:
	cmake -S src_cli -B build/build-cli -DCMAKE_BUILD_TYPE=Release
	cmake --build build/build-cli -- -j16

cli-debug:
	cmake -S src_cli -B build/build-cli -DCMAKE_BUILD_TYPE=Debug
	cmake --build build/build-cli -- -j16

clean-all:
	rm -rf build

clean-cli:
	rm -rf build/build-cli

demo testing [Asset Importer](http://www.assimp.org/)

build instructions

```
mkdir build
cd build
cmake ..
make -j8
ln -s ../corgi.stl ./
./demo
```

Demo code from [Open Asset Import Library Documentation](http://www.assimp.org/lib_html/usage.html)

Corgi STL from [Thingiverse](http://www.thingiverse.com/thing:462823)

Command line converter available on the CS system. See `assimp help`

Want to include this in your project? Just add `-lassimp` to your target_link_libraries command or to your list of external libraries in your CMakeLists.txt



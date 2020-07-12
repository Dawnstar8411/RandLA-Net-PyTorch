cd ../utils/nearest_neighbors
python3 setup.py install --home="."

cd ../cpp_wrappers/cpp_subsampling
python3 setup.py build_ext --inplace
cd ..
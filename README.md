# Usage

Protocol parameters should be given in .json format. They can be found under "protocols/" directory. Desired protocols should be input in the sequence design scripts. Currently there are 2 scripts:

- MultiEchoSpinEcho2D.py 
Implements a 2D MESE sequence to be used for T2 mapping.
- VariableFlipAngleGRE3D.py
Implements a multi flip angle 3D Gre sequence to be used for B1 and T1 mapping.

## Required Python Packages:

Non-exhaustive package list:

- Pypulseq (dev branch)
- Sigpy (required by pypulseq)
- Matplotlib
- Numpy
- Mplcursors (optional)

Dependencies are tested with conda. Environment exhaustive package list:

```
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main  
_openmp_mutex             5.1                       1_gnu  
backcall                  0.2.0                    pypi_0    pypi
blas                      1.0                         mkl  
brotli                    1.0.9                h5eee18b_7  
brotli-bin                1.0.9                h5eee18b_7  
bzip2                     1.0.8                h7b6447c_0  
ca-certificates           2022.07.19           h06a4308_0  
certifi                   2022.6.15       py310h06a4308_0  
cloudpickle               2.1.0                    pypi_0    pypi
cycler                    0.11.0             pyhd3eb1b0_0  
dbus                      1.13.18              hb2f20db_0  
debugpy                   1.6.3                    pypi_0    pypi
decorator                 5.1.1                    pypi_0    pypi
entrypoints               0.4                      pypi_0    pypi
expat                     2.4.4                h295c915_0  
fontconfig                2.13.1               h6c09931_0  
fonttools                 4.25.0             pyhd3eb1b0_0  
freetype                  2.11.0               h70c0345_0  
giflib                    5.2.1                h7b6447c_0  
glib                      2.69.1               h4ff587b_1  
gst-plugins-base          1.14.0               h8213a91_2  
gstreamer                 1.14.0               h28cd5cc_2  
icu                       58.2                 he6710b0_3  
intel-openmp              2021.4.0          h06a4308_3561  
ipykernel                 6.15.1                   pypi_0    pypi
ipython                   7.34.0                   pypi_0    pypi
jedi                      0.18.1                   pypi_0    pypi
jpeg                      9e                   h7f8727e_0  
jupyter-client            7.3.4                    pypi_0    pypi
jupyter-core              4.11.1                   pypi_0    pypi
kiwisolver                1.4.2           py310h295c915_0  
krb5                      1.19.2               hac12032_0  
lcms2                     2.12                 h3be6417_0  
ld_impl_linux-64          2.38                 h1181459_1  
lerc                      3.0                  h295c915_0  
libbrotlicommon           1.0.9                h5eee18b_7  
libbrotlidec              1.0.9                h5eee18b_7  
libbrotlienc              1.0.9                h5eee18b_7  
libclang                  10.0.1          default_hb85057a_2  
libdeflate                1.8                  h7f8727e_5  
libedit                   3.1.20210910         h7f8727e_0  
libevent                  2.1.12               h8f2d780_0  
libffi                    3.3                  he6710b0_2  
libgcc-ng                 11.2.0               h1234567_1  
libgfortran-ng            7.5.0               ha8ba4b0_17  
libgfortran4              7.5.0               ha8ba4b0_17  
libgomp                   11.2.0               h1234567_1  
libllvm10                 10.0.1               hbcb73fb_5  
libpng                    1.6.37               hbc83047_0  
libpq                     12.9                 h16c4e8d_3  
libstdcxx-ng              11.2.0               h1234567_1  
libtiff                   4.4.0                hecacb30_0  
libuuid                   1.0.3                h7f8727e_2  
libwebp                   1.2.2                h55f646e_0  
libwebp-base              1.2.2                h7f8727e_0  
libxcb                    1.15                 h7f8727e_0  
libxkbcommon              1.0.1                hfa300c1_0  
libxml2                   2.9.14               h74e7548_0  
libxslt                   1.1.35               h4e12654_0  
lz4-c                     1.9.3                h295c915_1  
matplotlib                3.5.2                    pypi_0    pypi
matplotlib-inline         0.1.6                    pypi_0    pypi
mkl                       2021.4.0           h06a4308_640  
mkl-service               2.4.0           py310h7f8727e_0  
mkl_fft                   1.3.1           py310hd6ae3a3_0  
mkl_random                1.2.2           py310h00e6091_0  
munkres                   1.1.4                      py_0  
ncurses                   6.3                  h5eee18b_3  
nest-asyncio              1.5.5                    pypi_0    pypi
nspr                      4.33                 h295c915_0  
nss                       3.74                 h0370c37_0  
numpy                     1.23.1          py310h1794996_0  
numpy-base                1.23.1          py310hcba007f_0  
openssl                   1.1.1q               h7f8727e_0  
packaging                 21.3               pyhd3eb1b0_0  
parso                     0.8.3                    pypi_0    pypi
pcre                      8.45                 h295c915_0  
pexpect                   4.8.0                    pypi_0    pypi
pickleshare               0.7.5                    pypi_0    pypi
pillow                    9.2.0           py310hace64e9_1  
pip                       22.1.2          py310h06a4308_0  
ply                       3.11            py310h06a4308_0  
prompt-toolkit            3.0.30                   pypi_0    pypi
psutil                    5.9.1                    pypi_0    pypi
ptyprocess                0.7.0                    pypi_0    pypi
pygments                  2.13.0                   pypi_0    pypi
pyparsing                 3.0.4              pyhd3eb1b0_0  
pypulseq                  1.4.0                    pypi_0    pypi
pyqt                      5.15.7          py310h6a678d5_1  
pyqt5-sip                 12.11.0                  pypi_0    pypi
pyqtgraph                 0.11.0                     py_0  
python                    3.10.4               h12debd9_0  
python-dateutil           2.8.2              pyhd3eb1b0_0  
pyzmq                     23.2.1                   pypi_0    pypi
qt-main                   5.15.2               h327a75a_7  
qt-webengine              5.15.9               hd2b0992_4  
qtwebkit                  5.212                h4eab89a_4  
readline                  8.1.2                h7f8727e_1  
scipy                     1.8.1                    pypi_0    pypi
setuptools                63.4.1          py310h06a4308_0  
sip                       6.6.2           py310h6a678d5_0  
six                       1.16.0             pyhd3eb1b0_1  
spyder-kernels            2.3.2                    pypi_0    pypi
sqlite                    3.39.2               h5082296_0  
tk                        8.6.12               h1ccaba5_0  
toml                      0.10.2             pyhd3eb1b0_0  
tornado                   6.1             py310h7f8727e_0  
traitlets                 5.3.0                    pypi_0    pypi
tzdata                    2022a                hda174b7_0  
wcwidth                   0.2.5                    pypi_0    pypi
wheel                     0.37.1             pyhd3eb1b0_0  
wurlitzer                 3.0.2                    pypi_0    pypi
xz                        5.2.5                h7f8727e_1  
zlib                      1.2.12               h7f8727e_2  
zstd                      1.5.2                ha4553b6_0  
```
# 6369 Mini PC Vision

## Coral PCIe Driver for Linux 6.4+
The gasket driver in Google's repo dosen't install on Linux kernel 6.4 and higher.

Instead you can build the driver from source

* Clone the gasket-driver from https://github.com/google/gasket-driver/
* Install build dependencies `sudo apt install devscripts dh-dkms build-essential`
* Build `debuild -us -uc -tc -b`
* Install `cd .. && dpkg -i gasket-dkms_1.0-18_all.deb`
* Finish setting up Coral from official instructions


## System Dependencies

 * `libdbus-1-dev`
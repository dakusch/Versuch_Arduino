sudo nano /etc/udev/rules.d/99-rtlsdr.rules

# RTL-SDR Radio
SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2838", MODE="0666"

sudo udevadm control --reload-rules
sudo udevadm trigger

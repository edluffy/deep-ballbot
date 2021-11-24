#include "Arduino.h"
#include <SimpleFOC.h>


MagneticSensorSPIConfig_s config = {
  .spi_mode = SPI_MODE1,
  .clock_speed = 1000000,
  .bit_resolution = 14,
  .angle_register = 0x3FFF,
  .data_start_bit = 13,
  .command_rw_bit = 14,
  .command_parity_bit = 15
};

int chip_select = 10;

MagneticSensorSPI sensor = MagneticSensorSPI(config, chip_select);

void setup() {
  Serial.begin(9600);
  sensor.init();
  Serial.println("Sensor ready");
  _delay(1000);
}

void loop() {
  sensor.update();
  Serial.print(sensor.getAngle(), 22);
  Serial.print("\t");
  Serial.println(sensor.getVelocity(), 23);
}

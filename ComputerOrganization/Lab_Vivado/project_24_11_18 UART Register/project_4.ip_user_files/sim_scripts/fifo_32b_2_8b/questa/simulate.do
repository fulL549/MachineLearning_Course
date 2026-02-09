onbreak {quit -f}
onerror {quit -f}

vsim -lib xil_defaultlib fifo_32b_2_8b_opt

do {wave.do}

view wave
view structure
view signals

do {fifo_32b_2_8b.udo}

run -all

quit -force

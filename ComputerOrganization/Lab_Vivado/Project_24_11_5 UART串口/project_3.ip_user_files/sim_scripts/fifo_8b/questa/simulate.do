onbreak {quit -f}
onerror {quit -f}

vsim -lib xil_defaultlib fifo_8b_opt

do {wave.do}

view wave
view structure
view signals

do {fifo_8b.udo}

run -all

quit -force

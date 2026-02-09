onbreak {quit -f}
onerror {quit -f}

vsim -lib xil_defaultlib u2r_cmd_fifo_opt

do {wave.do}

view wave
view structure
view signals

do {u2r_cmd_fifo.udo}

run -all

quit -force

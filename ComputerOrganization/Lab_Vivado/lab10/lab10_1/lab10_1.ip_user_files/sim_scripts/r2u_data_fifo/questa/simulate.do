onbreak {quit -f}
onerror {quit -f}

vsim -lib xil_defaultlib r2u_data_fifo_opt

do {wave.do}

view wave
view structure
view signals

do {r2u_data_fifo.udo}

run -all

quit -force

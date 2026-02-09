onbreak {quit -f}
onerror {quit -f}

vsim -voptargs="+acc" -L xpm -L unisims_ver -L unimacro_ver -L secureip -lib xil_defaultlib xil_defaultlib.fifo_32b_2_8b xil_defaultlib.glbl

do {wave.do}

view wave
view structure
view signals

do {fifo_32b_2_8b.udo}

run -all

quit -force

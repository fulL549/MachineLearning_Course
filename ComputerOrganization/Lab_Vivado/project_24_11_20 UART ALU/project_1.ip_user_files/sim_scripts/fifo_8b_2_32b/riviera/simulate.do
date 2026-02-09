onbreak {quit -force}
onerror {quit -force}

asim +access +r +m+fifo_8b_2_32b -L xpm -L fifo_generator_v13_2_5 -L xil_defaultlib -L unisims_ver -L unimacro_ver -L secureip -O5 xil_defaultlib.fifo_8b_2_32b xil_defaultlib.glbl

do {wave.do}

view wave
view structure

do {fifo_8b_2_32b.udo}

run -all

endsim

quit -force

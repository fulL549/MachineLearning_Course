onbreak {quit -f}
onerror {quit -f}

vsim -lib xil_defaultlib data_ram_opt

do {wave.do}

view wave
view structure
view signals

do {data_ram.udo}

run -all

quit -force

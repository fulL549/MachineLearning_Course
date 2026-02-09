onbreak {quit -f}
onerror {quit -f}

vsim -lib xil_defaultlib datadata_opt

do {wave.do}

view wave
view structure
view signals

do {datadata.udo}

run -all

quit -force

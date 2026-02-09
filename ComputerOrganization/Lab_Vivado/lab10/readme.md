# 实验10 单周期处理器应用实验 代码模板

文件目录：  
```
lab10_code_template    
|-proc_top.v    //调用了MIPS单周期处理器顶层文件放在\MIPS_core\rtl\top.v，将top信号导入到sys_monitor中。
|-readme.md   
|-\MIPS_core    //包含了MIPS单周期处理器的文件 
|-\uart_monitor //包含了串口监视器的文件 
 
```

## 需要引入的文件

MIPS单周期处理器全部放在`\MIPS_core\rtl`下面
MIPS单周期处理器顶层文件放在`\MIPS_core\rtl\top.v`里面
串口监视器文件全部放在`\uart_monitor`下面
要自己看下代码结构

## 需要修改的文件

根据实验手册完成MIPS_CORE。  
MIPS_CORE可以从Lab9中粘贴已经完成的代码，不一定要完全按照本次的模板。串口监视器部分如果不了解内容不要随意修改。

```
|-proc_top.v  //调用了MIPS单周期处理器顶层文件放在\MIPS_core\rtl\top.v，将top信号导入到sys_monitor中。根据显示信号的需要可以自己配置
```

整个处理器的顶层模块，与 Lab8/9 的修改方式类似。
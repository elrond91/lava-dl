#!/usr/bin/python
import jtop 

if __name__ == '__main__':                
    with jtop.jtop() as jetson:
        print(jetson.power, jetson.gpu, jetson.memory)
        
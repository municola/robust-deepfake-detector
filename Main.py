#!/usr/bin/env python
# -*- coding: utf-8 -*-
import polimi.gan_vs_real_detector as gan_vs_real_detector

def main():
    print("before ")
    gan_vs_real_detector(img_path='dataTest/real/pic.png')
    print("after")
    return

if __name__ == "__main__":
    main()

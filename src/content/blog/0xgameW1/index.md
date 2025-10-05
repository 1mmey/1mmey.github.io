---
title: '0xGame'
publishDate: 2025-02-09
updatedDate: 2025-02-24
description: '3D imagery has the power to bring cinematic visions to life and help accurately plan tomorrow’s cityscapes. Here, 3D expert Ricardo Ortiz explains how it works.'
tags:
  - Example
  - 3D
language: 'English'
heroImage: { src: './thumbnail.jpg', color: '#D58388' }
---

## 一、BabyBase

从check_flag函数可以发现base64字符串，cyberchef可以解出flag

0xGame{N0w_y0u_kn0w_B4se64_Enc0d1ng_w3ll!}

## 二、BinaryMaster

直接用IDA打开即可获得flag

0xGame{114514cc-a3a7-4e36-8db1-5f224b776271}

## 三、SignSign

shift+F12检索字符串即可找到flag的另一半

## 四、Xor-Beginning

exp:

```python
cipher = "~5\v*',3"
v5 = [0] * 30
for i in range(len(cipher)):
    v5[i] = ord(cipher[i])
v5[7] = 31
v5[8] = 118
v5[9] = 55
v5[10] = 27
v5[11] = 114
v5[12] = 49
v5[13] = 30
v5[14] = 54
v5[15] = 12
v5[16] = 76
v5[17] = 68
v5[18] = 99
v5[19] = 114
v5[20] = 87
v5[21] = 73
v5[22] = 8
v5[23] = 69
v5[24] = 66
v5[25] = 1
v5[26] = 90
v5[27] = 4
v5[28] = 19
v5[29] = 76
flag = ''
for i in range(len(v5)):
    flag += chr(v5[i] ^ (78-i))
print(flag)
#0xGame{X0r_1s_v3ry_Imp0rt4n7!}
```

## 五、Xor-Endian

与beginning不同是要考虑大小端序的问题

exp:

```python
key = "Key0xGame2024"
flag = ''
v6 =[0x7b,0x1d,0x3e,0x51,0x15,0x22,0x1a,0xf,0x56,0xa,0x51,0x56,0x0,0x28,0x5d,0x54,0x7,0x4b,0x74,0x5,0x40,0x51,0x54,0x8,0x54,0x19,0x72,0x56,0x1d,0x4,0x55,0x76,0x56,0xb,0x54,0x57,0x7,0xb,0x55,0x73,0x1,0x4f,0x8,0x5]
for i in range(len(v6)):
    flag += chr(v6[i]^ord(key[i%13]))
print(flag)
#0xGame{b38ad4c8-733d-4f8f-93d4-17f1e79a8d68}
```

# LinAgent
## AI System Agent For Linux

An intelligent agent built for performing automation tasks and executing scripts, system commands for linux.

Data Structure
```
├── data
│   └── system.json
├── LICENSE
├── main.py
├── README.md
└── test
    └── test_api.py
```

I wrote system commands in `system.json` as a dataset that contains window management, software management and file management for KDE Plasma 6 for now, will add cross-DE support in the future. 


This project uses Gemini 2.5 Flash Lite API for now, will add local AI support after training a low-hardware required model like Phi3 for system management commands in an efficiently way.


You can give commands to LinAgent with a prompt for now, will add text-to-speech and speech-to-text in the future for easier use.


Will add the app as a system service so that we won't need to open the app everytime in the future.

Will implement `OCR` and `ydotool` in the future to perform in-app processes too.



<img width="1894" height="520" alt="image" src="https://github.com/user-attachments/assets/b56ca10b-eb7a-42a4-bd81-df17a9b4e97b" />

# Edge-assisted Mobile AR (Python Code for the Edge Server)

# Description

We designed and implemented an edge-based mobile AR system to analyze the interactions between `AR configurations` and the mobile device's `energy consumption`. The edge server is developed to process received image frames and to send the detection results back to the Mobile AR client. Two major modules are implemented on the edge server: (i) the `communication handler` which establishes a `TCP socket` connection with the Mobile AR device and (ii) the `analytics handler` which performs object detection for the Mobile AR client. The analytics handler is designed based on a custom framework called Darknet with GPU acceleration and runs YOLOv3, a large Convolutional Neural Networks (CNN) model.

# Citation
`If you use the code in your work please cite our papers!`

* Haoxin Wang and Jiang Xie, "User Preference Based Energy-Aware Mobile AR System with Edge Computing," in *Proc. IEEE International Conference on Computer Communications (INFOCOM 2020),* Beijing, China, Apr. 2020, pp.1-10. (https://infocom2020.ieee-infocom.org/accepted-paper-list-main-conference)

* Haoxin Wang, BaekGyu Kim, Jiang Xie, and Zhu Han, "How is Energy Consumed in Smartphone Deep Learning Apps? Executing Locally vs. Remotely," in *Proc. IEEE Global Communications Conference (GLOBECOM 2019),* Waikoloa, HI, Dec. 2019, pp.1-6. (https://doi.org/10.1109/GLOBECOM38437.2019.9013647)

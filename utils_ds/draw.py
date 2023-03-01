import numpy as np
import cv2

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

NAMES = ['Tie','Boots','Shirt','Jeans','Suit','Dhoti','Long dress/gown','Hoodie','Pants','Sherwani','Turban','Sunglasses & goggles','Spectacles & glasses','Hand bag','Necklace','Hat','Scarf','Backpack','Cap','Sneakers','Wrist watch','Salwar Suit','Earring','Anklet','Bangle','Barefoot','Bracelet','Chappals','Nose ring','Ring','Shorts','Denim jacket','Kurta\kurti','Leggins','Mask','Pyjama','Sweater','Taqiyah','Heels','Waist Coat','Night Gown','Long skirt','Short skirt','Headphones','Belt','Duffle bags','Earphones','Smart watch','Suitcase','Swimwear','Sling Bag']

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None,classes=None, offset=(0,0)):
    for i,box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        cls= NAMES[int(classes[i])]
        print(cls)
        color = compute_color_for_labels(id)
        label = str(id)+"_"+cls#'{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
        cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
    return img

    def draw(self, img, boxinfo):
        for xyxy, conf, cls in boxinfo:
            
            score = f"{conf:.2f}"
            label_text = f"{self.classes[int(cls)]} {score}"
            self.plot_one_box(xyxy, img, label=label_text, color=self.colors[int(cls)], line_thickness=2)
        cv2.imshow('Press ESC to Exit', img) 
        cv2.waitKey(1)



if __name__ == '__main__':
    for i in range(82):
        print(compute_color_for_labels(i))

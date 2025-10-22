import cv2
import numpy as np
from matplotlib import pyplot as plt



img_color=cv2.imread(r"C:\Users\User\.spyder-py3\project\6.jpg")
img_color=cv2.cvtColor(img_color,cv2.COLOR_BGR2RGB)

img_bgr=cv2.cvtColor(img_color,cv2.COLOR_RGB2BGR)
cells=img_bgr[:, :, 0]



def histogram_equalization(image):
    hist=cv2.calcHist([image],[0],None,[256],[0, 256]).flatten()
    pdf=hist/hist.sum()
    cdf=pdf.cumsum()
    sk=np.round(cdf*255).astype(np.uint8)
    equalized=cv2.LUT(image,sk)
    return equalized


def manual_CLAHE(image,tile_size=(8,8),clip_limit=2.0):
    L=256
    img_h,img_w=image.shape
    tile_h,tile_w=tile_size
    n_tiles_h=img_h//tile_h
    n_tiles_w=img_w//tile_w

    output=np.zeros_like(image,dtype=np.uint8)

    for i in range(n_tiles_h):
        for j in range(n_tiles_w):
            y0,y1=i*tile_h,(i+1)*tile_h
            x0,x1=j*tile_w,(j+1)*tile_w
            tile=image[y0:y1,x0:x1]

            hist=cv2.calcHist([tile],[0],None,[256],[0,256]).flatten()

            max_clip=clip_limit*tile.size/L
            excess=hist-max_clip
            excess[excess<0]=0
            hist=np.minimum(hist,max_clip)

            redistribute=excess.sum()/L
            hist+=redistribute

            pdf=hist/hist.sum()
            cdf=pdf.cumsum()
            sk=np.round(cdf*(L-1)).astype(np.uint8)

            output[y0:y1,x0:x1]=cv2.LUT(tile,sk)

    return output



cv2.imshow("Original Image",cells)
cv2.imwrite('step1_original.jpg',cells)


manual_he=histogram_equalization(cells)
cv2.imshow("Manual Histogram Equalization",manual_he)
cv2.imwrite('step1_manual_histogram_equalization.jpg',manual_he)


manual_clahe=manual_CLAHE(cells,tile_size=(8, 8),clip_limit=2.0)
cv2.imshow("Manual CLAHE",manual_clahe)
cv2.imwrite('step1_manual_clahe.jpg',manual_clahe)


clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8, 8))
opencv_clahe=clahe.apply(cells)
cv2.imshow("OpenCV Built-in CLAHE",opencv_clahe)
cv2.imwrite('step1_opencv_clahe.jpg',opencv_clahe)

plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)
plt.imshow(cells, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(manual_he, cmap='gray')
plt.title("Manual Histogram Equalization")
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(manual_clahe, cmap='gray')
plt.title("Manual CLAHE")
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(opencv_clahe, cmap='gray')
plt.title("OpenCV CLAHE")
plt.axis('off')

plt.tight_layout()
plt.savefig('step1_equalization_comparison.jpg', dpi=150, bbox_inches='tight')
plt.show()

equalized_cells=manual_clahe


def gaussian_kernel(sigma):
    size=int(5 * sigma) | 1
    k=size//2
    kernel=np.zeros((size, size),dtype=np.float64)
    for i in range(size):
        for j in range(size):
            x,y=i-k,j-k
            kernel[i,j]=(1/(2*np.pi*sigma**2))*np.exp(-(x**2+y**2)/(2*sigma**2))
    kernel/=np.sum(kernel)
    return kernel


def apply_convolution(image,kernel):
    img_h,img_w=image.shape
    k_h,k_w =kernel.shape
    pad_h,pad_w=k_h//2,k_w//2
    padded=np.pad(image,((pad_h,pad_h),(pad_w,pad_w)),mode='reflect')
    output=np.zeros_like(image,dtype=np.float64)
    kernel=np.flip(kernel)
    for i in range(img_h):
        for j in range(img_w):
            s=0.0
            for ki in range(k_h):
                for kj in range(k_w):
                    s+=padded[i +ki,j+kj]*kernel[ki,kj]
            if s<0:
                s=0
            elif s>255:
                s=255
            output[i,j]=s
    return output.astype(np.uint8)


kernel=gaussian_kernel(1.0)
blurred_cells=apply_convolution(equalized_cells,kernel)


cv2.imshow("Blurred Cells ",blurred_cells)
cv2.imwrite('step2_blurred_cells.jpg', blurred_cells)




def manual_otsu_threshold(img):
    hist=cv2.calcHist([img],[0],None,[256],[0, 256]).flatten()

    total=img.size
    
    n_i=hist
    P_i=n_i/total
    
    m_g=0.0
    for intensity in range(256):
        m_g+=intensity*P_i[intensity]
    
    max_sigma2_B=0
    threshold=0
    
    for i in range(256):
        P1=0.0
        for j in range(i+1):
            P1+=P_i[j]
        
        if P1==0 or P1==1:
            continue
        
        m=0.0
        for j in range(i+1):
            m+=j*P_i[j]
        
        sigma2_B=((m_g*P1-m)**2)/(P1*(1-P1))
        
        if sigma2_B>max_sigma2_B:
            max_sigma2_B=sigma2_B
            threshold=i
    
    binary=np.zeros_like(img, dtype=np.uint8)
    binary[img>threshold]=255
    return threshold,binary


otsu_thresh,thresh_manual=manual_otsu_threshold(blurred_cells)
print(f"\nManual Otsu Threshold: {otsu_thresh}")
cv2.imshow("Manual Otsu Threshold",thresh_manual)
cv2.imwrite('step3_thresh_manual.jpg', thresh_manual)


ret1,thresh=cv2.threshold(blurred_cells,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(f"Otsu Threshold: {ret1}")
cv2.imshow("Otsu Threshold",thresh)
cv2.imwrite('thresh.jpg', thresh)

plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)
plt.imshow(thresh_manual, cmap='gray')
plt.title("Manual Otsu")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(thresh, cmap='gray')
plt.title("built-in otsu")
plt.axis('off')

plt.tight_layout()
plt.show()


se=np.ones((3,3),np.uint8)

eroded=cv2.erode(thresh_manual,se,iterations=3)
cv2.imshow("Eroded image",eroded)
cv2.imwrite('step4_eroded.jpg', eroded)

opening=cv2.dilate(eroded,se,iterations=3)
cv2.imshow("opening(Erosion+Dilation)",opening)
cv2.imwrite('step4_opening.jpg', opening)

dilated=cv2.dilate(opening,se,iterations=3)
cv2.imshow("Dilated",dilated)
cv2.imwrite('step4_dilated.jpg', dilated)

closing=cv2.erode(dilated,se,iterations=3)
cv2.imshow("CLosing(DIlation+Erosion",closing)
cv2.imwrite('step4_closing.jpg', closing)



eroded=cv2.erode(closing,se,iterations=1)
boundaries=closing-eroded

impose=img_color
for i in range(impose.shape[0]):
    for j in range(impose.shape[1]):
        if boundaries[i,j]>0:
            impose[i, j]=[255,0,0]


cv2.imshow("Segmented Borders on Original Image",cv2.cvtColor(impose,cv2.COLOR_RGB2BGR))
cv2.imwrite('step4_overlay_borders.jpg', cv2.cvtColor(impose, cv2.COLOR_RGB2BGR))


def get_count_nuclei(binary_img):
    binary=np.zeros_like(binary_img,dtype=int)
    for i in range(binary_img.shape[0]):
        for j in range(binary_img.shape[1]):
            if binary_img[i,j]>0:
                binary[i,j]=1
    
    rows,cols=binary.shape
    labels=np.zeros_like(binary,dtype=int)
    current_label=1
    
    neighbors=[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    
    def nei_label(r,c,label):
        stack=[(r,c)]
        while stack:
            x,y=stack.pop()
            if (0<=x<rows) and (0<=y<cols):
                if binary[x,y]==1 and labels[x,y]==0:
                    labels[x,y]=label
                    for dx,dy in neighbors:
                        stack.append((x+dx,y+dy))
    
    for i in range(rows):
        for j in range(cols):
            if binary[i,j]==1 and labels[i,j]==0:
                nei_label(i,j,current_label)
                current_label+=1
    
    count=current_label-1
    return labels,count


labels,count=get_count_nuclei(boundaries)
print(f"\ncount: {count}")


count2,labels2=cv2.connectedComponents(closing,connectivity=8) 

print(f"built-in count: {count2}")
print(f"\nDifference: {abs(count-count2)}")

cv2.waitKey(0)
cv2.destroyAllWindows()

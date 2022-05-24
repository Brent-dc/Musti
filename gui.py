from tkinter import * 
from PIL import ImageTk, Image
from bs4 import BeautifulSoup as bs
import pickle
import requests
from fitter import fit
from requests import request
from procesImage import procesImage
import time
panel = ""
main_window = Tk("cat detector")

ovr =  pickle.load(open("ensemble.mod", 'rb'))
#gui interaction

def on_click():
    if(url_cat.get() == "https://musti-24d3d.web.app"):
            while True:
                url = "https://musti-24d3d.web.app"
                l1 = Label(main_window, text = "ok!")
                l1.pack()
                page = requests.get('https://musti-24d3d.web.app')
                soup = bs(page.text, features="lxml")
                tag=soup.find('img')
                timestamp = tag.get('alt')
                print(soup)
                img = requests.get(f"{url}/pic.jpg")
                with  open(f"file.jpg", 'wb') as file:
                                file.write(img.content)

                                img = Image.open("file.jpg")
                        
                                file.close()
                print(img)
                img_g = ImageTk.PhotoImage(img)
                panel = Label(main_window, image = img_g)
                panel.pack(fill = "both")
                
                print(img)
                res = procesImage(img, timestamp, ovr) 
                Label(main_window, text = res).pack()

                main_window.mainloop()
                time.sleep(5)            
    else: 
            
            l1 =  Label(main_window, text = "nothing found !")
            l1.pack()
            main_window.update()



#labels
Label(main_window, text = "cat url").pack()

#input
url_cat = Entry(main_window, width = 50)
url_cat.pack()


Button(main_window , text ="Verify", command = on_click).pack()
path_pics = Entry(main_window, width = 50)
path_pics.pack()
def train():
        fit(path_pics.get())
Button(main_window , text ="Upload labeled cat pics", command = train).pack()


main_window.mainloop()


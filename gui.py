from tkinter import * 
from PIL import ImageTk, Image
from bs4 import BeautifulSoup as BS
import pickle
from procesImage import procesImage
panel = ""
main_window = Tk("cat detector")
test_url = r'C:\Users\brent\OneDrive\Bureaublad\classificatie\aanwezig\20220207_123451'
print(test_url)
ovr =  pickle.load(open("oneVsRest.mod", 'rb'))
#gui interaction
def on_click():
    if(url_cat.get() == "https://ibb.co/p0fgwGG"):
            l1 = Label(main_window, text = "ok!")
            l1.pack()
            img = Image.open(r"ml_gui\pic\20220210_095103.jpg")
            img = ImageTk.PhotoImage(img)
            panel = Label(main_window, image = img)
            panel.pack(fill = "both")
         
            print(img)
            res = procesImage(Image.open(r"ml_gui\pic\20220210_095103.jpg"), "20220210_095103.jpg", ovr) 
            Label(main_window, text = res).pack()

            main_window.mainloop()
            
    else: 
            
            l1 =  Label(main_window, text = "nothing found !")
            l1.pack()
            main_window.update()



#labels
Label(main_window, text = "cat url").pack()

#input
url_cat = Entry(main_window, width = 50)
url_cat.pack()

#buttons
Button(main_window , text ="Verify", command = on_click).pack()
Button(main_window , text ="Upload labeled cat pics", command = on_click).pack()





main_window.mainloop()


import taipy as  tp 
import taipy.gui.builder as tgb
from taipy.gui import Icon
from taipy import Config

with tgb.Page() as page:
    tgb.image("images/icons/logo.png")
    tgb.text("S&P stock value over time", mode = "md")
    
if __name__ == "__main__":
    gui = tp.Gui(page)
    gui.run(title="S&P Stock Value", use_reloader=True)
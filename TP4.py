import tkinter as tk
import tkinter.messagebox as tkmsg
from PIL import ImageGrab

"""
ean 13

2 - 3 first digits --> national GS1

4 - 6 digits       --> company number

2 - 6 digits       --> item reference

1 last digit       --> checksum

Indonesian GS1 : 899

Encoding EAN13
3 groups split

first group : mysterious digit (sedikit diluar barcode)
### some kind of selector?
0 : standard UPC

second group: first 6 digit
third group : last 6 digit

each digit occupies 7 bit

section of EAN13
SXXXXXXMRRRRRRE

S: start    --> 101
M: middle   --> 01010
E: end      --> 101
S, M and E is a little bit longer than ordinary digit and called guard bars

X and R are digits with total of 95 bits

EAN 13 dibuat dengan sengaja agar tiap digit hanya memiliki 2 bar dan 2 space

S, M dan E memiliki 2 bar dan tiap digit memiliki 2 bar
total bar = 30 bar

last special digit calculated from multiplying digit with its weight and do (10 - x) % 10

weight digits

1   2   3   4   5   6   7   8   9  10  11  12
1   3   1   3   1   3   1   3   1   3   1   3

8  9  0  9  0  9  <-- kali 1
9  7  2  8  9  7  <-- kali 3

"""


# dict yang mengubah dari format dan digit menjadi binary 7 bit
dict_ean_code = {
    "L": {
        "0": "0001101",
        "1": "0011001",
        "2": "0010011",
        "3": "0111101",
        "4": "0100011",
        "5": "0110001",
        "6": "0101111",
        "7": "0111011",
        "8": "0110111",
        "9": "0001011",
    },
    "G": {
        "0": "0100111",
        "1": "0110011",
        "2": "0011011",
        "3": "0100001",
        "4": "0011101",
        "5": "0111001",
        "6": "0000101",
        "7": "0010001",
        "8": "0001001",
        "9": "0010111",
    },
    "R": {
        "0": "1110010",
        "1": "1100110",
        "2": "1101100",
        "3": "1000010",
        "4": "1011100",
        "5": "1001110",
        "6": "1010000",
        "7": "1000100",
        "8": "1001000",
        "9": "1110100",
    }
}

# untuk menentukan format yang digunakan yang ditentukan dari digit pertama
dict_type = {
    "0": "LLLLLLRRRRRR",
    "1": "LLGLGGRRRRRR",
    "2": "LLGGLGRRRRRR",
    "3": "LLGGGLRRRRRR",
    "4": "LGLLGGRRRRRR",
    "5": "LGGLLGRRRRRR",
    "6": "LGGGLLRRRRRR",
    "7": "LGLGLGRRRRRR",
    "8": "LGLGGLRRRRRR",
    "9": "LGGLGLRRRRRR",
}

class MainWindow(tk.Frame):
    """Main window"""
    def __init__(self, master = None):
        super().__init__(master)
        self.pack()

        self.file_name_var = tk.StringVar()
        self.ean_code_var = tk.StringVar()

        self.create_widgets()


    def create_widgets(self):
        self.file_name_label = tk.Label(self,
                                        text = 'Save barcode to PS file [eg:EAN13.eps]:',
                                        justify='center',
                                        font=('Arial', 20, 'bold'))
        self.file_name_entry = tk.Entry(self,
                                        textvariable=self.file_name_var)
        self.ean_code_label = tk.Label(self,
                                       text = 'Enter code (first 12 decimal digits):',
                                       justify='center',
                                       font=('Arial', 20, 'bold'))
        self.ean_code_entry = tk.Entry(self,
                                       textvariable=self.ean_code_var)
        self.btn_generate = tk.Button(self,
                                      text = "generate!",
                                      command = self.check_validasi)
        self.btn_exit = tk.Button(self,
                                  text = "EXIT",
                                  command = self.quit)
        self.canvas = EAN13_Canvas(master=self)

        self.file_name_label.pack()
        self.file_name_entry.pack()
        self.ean_code_label.pack()
        self.ean_code_entry.pack()
        self.canvas.pack()
        self.btn_generate.pack()
        self.btn_exit.pack()

    def check_validasi(self):
        """
        Melakukan validasi terhadap nama file dan code dari EAN-13
        """
        # debug
        # print(self.file_name_var.get())
        # print(self.ean_code_var.get())
        # end of debug

        # cek file_name_var apakah ada illegal character
        for illegal_char in """#%&{}\\$!'":@<>*?/+`|=""":
            if illegal_char in self.file_name_var.get():
                errmsg = "A file name can't contain any of the following characters:\n"
                errmsg+= """#%&{}\\$!'":@<>*?/+`|="""
                self.popup_error_message(errmsg)
                return

        # cek file_name_var apakah berakhir dengan .ps
        if not self.file_name_var.get().endswith('.eps'):
            self.popup_error_message("Please make the file name endswith .eps")
            return

        # cek ean_code_var apakah memiliki 12 char dan hanya angka
        if len(self.ean_code_var.get()) != 12 or not self.ean_code_var.get().isdecimal():
            self.popup_error_message("Please enter correct input code.")
            return

        self.canvas.delete('all')
        self.canvas.string = self.ean_code_var.get()
        self.after(1000, self.canvas._snapCanvas())

    def popup_error_message(self, errmsg):
        """For popup error message"""
        tkmsg.showerror(title="Wrong Input!",
                        message=errmsg)

    # https://stackoverflow.com/a/41965743
    

class EAN13_Canvas(tk.Canvas):
    """canvas untuk menggambar ean13 barcode"""
    def __init__(self, ean13_string = "", master = None):
        super().__init__(master, bg='white', height=350, width=300)
        self.string = ean13_string

    @property
    def string(self):
        return self._string

    @string.setter # trigger ada perubahan nilai di property string
    def string(self, ean13_string):
        self._string = ean13_string
        if self._string != '':
            self.process_ean_code(self._string)

    def process_ean_code(self, string):
        """
        process_ean_code
        ================
        mengolah 12 digit ean dari user menjadi gambar di canvas
        """
        x = 55
        y = 100
        step = 0
        width = 2
        height = 150
        i = 1

        # first_digit : digit pertama dari ean13
        # check_digit : digit terakhir dari ean13 (checksum)
        # ean_full_string : full 13 digit ean13
        # ean_bin_string  : ean_full_string yang diubah menjadi binary string (hanya index 1 sampai akhir)
        first_digit = string[0]
        check_digit = self.get_last_digit_ean13(string)
        ean_full_string = string + check_digit
        ean_bin_string = self.digits_to_ean13(ean_full_string[1:], dict_type[first_digit])

        # slip S, M and E (guard bars)
        ean_bin_string = ['101'] + ean_bin_string[0:6] + ['01010'] + ean_bin_string[6:] + ['101']
        print(ean_bin_string)
        
        # write first digit
        self.create_text(x - 8, y + height + 20,
                         font="Arial 16 bold",
                         text=str(ean_full_string[0]))

        # Drawing bars and write digit below it
        for ean_digit in ean_bin_string:
            if len(ean_digit) != 7:
                # for guard
                offset_guard = 10
                warna = "#2929ff"
            else:
                # for digits
                offset_guard = 0
                warna = "#008000"
                # offset
                self.create_text(x + step + 8, y + height + 20,
                         font="Arial 16 bold",
                         text=str(ean_full_string[i]))
                i += 1

            for bit in ean_digit:
                # make bars
                if bit == '1':
                    self.create_rectangle(x + step, y,
                                          x + step + width, y + height + offset_guard,
                                          fill=warna,
                                          width=0)
                else:
                    self.create_rectangle(x + step, y, x + step + width, y + height + offset_guard,
                                        fill="#FFF",
                                        width=0)
                step += width

        # Writing text
        # https://stackoverflow.com/a/17737103
        self.create_text(x + 95, y - 30,
                         font="Arial 16 bold",
                         text="EAN-13 Barcode:")
        # self.create_text(x + 85, y + height + 20,
        #                  font="Arial 16 bold",
        #                  text=f"{first_digit}  {''.join([str(a) for a in ean_full_string[:6]])}  {''.join([str(a) for a in ean_full_string[6:]])}")
        self.create_text(x + 95, y + height + 50,
                         fill="#fda605",
                         font="Arial 16 bold",
                         text=f"Check Digit: {check_digit}")

        self.pack()

    def _snapCanvas(self):
        """snap canvas and save it to a file"""
        x = self.winfo_rootx()
        y = self.winfo_rooty()
        x1= x + self.winfo_width()
        y1= y + self.winfo_height()

        box= (x, y, x1, y1) # coordinates of canvas
        ImageGrab.grab(bbox=box).save(self.master.file_name_var.get())

    def digits_to_ean13(self, string, format):
        """
        digits_to_ean13
        ===============
        Merubah dari string [0-9] menjadi binary ean encoded

        Parameter:
        ==========
        string:
            string yang berisi 12 char yang hanya berisi angka
        format:
            string yang berisi 12 char yang hanya berisi [ "L" | "G" | "R" ]

        Return:
        =======
        res:
            binary ean encoded list

        contoh:
        >>> digits_to_ean13("997029809979", "LGLGGLRRRRRR")
        ['0001011', '0010111', '0111011', '0100111', '0011011', '0001011', '1001000', '1110010', '1110100', '1110100', '1000100', '1110100']
        """
        res = []
        for i in range(len(string)):
            res.append(dict_ean_code[format[i]][string[i]])
        return res

    def get_last_digit_ean13(self, string):
        """
        get_last_digit_ean13
        ====================
        Menentukan digit terakhir dari ean code

        Parameter:
        ==========
        string:
            string yang berisi 12 char yang hanya berisi angka

        Return:
        =======
        res:
            last digit dari ean13

        contoh:
        >>> get_last_digit_ean13("899702980997")
        9
        """
        digits = [int(a) for a in string]
        total_sum = sum(digits[::2]) + sum(digits[1::2])*3
        print((10 - total_sum % 10) % 10)
        return str((10 - total_sum % 10) % 10)

def main():
    m = MainWindow()
    m.master.title("EAN-13 [by Shariyl]")
    m.master.mainloop()

if __name__ == "__main__":
    main()
#------------------------------- Global Imports -------------------------------#

import os, sys
from pathlib import Path

import logging
import queue
import yaml

import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from tkinter import filedialog

from src.adcs_modes import Production, Training

#------------------ Queue Logging Handler and Console Output ------------------#
# By: https://beenje.github.io/blog/posts/logging-to-a-tkinter-scrolledtext-widget
# Code: https://github.com/beenje/tkinter-logging-text-widget/blob/master/main.py

class QueueHandler(logging.Handler):
    """Class that inherits from Handler to send logging records to a queue

    It can be used from different threads and log from other modules. 
    The ConsoleUI class polls this queue to display records in a ScrolledText widget. 
    """

    def __init__(self, log_queue: queue.Queue) -> None:
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: str) -> None:
        self.log_queue.put(record)


class ConsoleUI:
    """Polls messages from a logging queue and displays them in a ScrolledText widget"""

    def __init__(self, frame, log_queue: queue.Queue) -> None:
        self.frame = frame
        self.log_queue = log_queue

        # Create ScrolledText widget
        self.scrolled_text = ScrolledText(self.frame, state='disabled', height=12)
        self.scrolled_text.grid(row=1, column=0, sticky='nsew')
        self.scrolled_text.configure(font='TkFixedFont')
        self.scrolled_text.tag_config('DEBUG', foreground='gray')
        self.scrolled_text.tag_config('INFO', foreground='black')
        self.scrolled_text.tag_config('WARNING', foreground='orange')
        self.scrolled_text.tag_config('ERROR', foreground='red')
        self.scrolled_text.tag_config('CRITICAL', foreground='red', underline=1)

        # Initialise QueueHandler with log_queue passed in from main()
        self.queue_handler = QueueHandler(self.log_queue)
        self.queue_handler.setFormatter(logging.Formatter(
            fmt='[%(levelname)s] %(asctime)s - %(message)s',
            datefmt='%d-%b-%Y %H:%M:%S',
        ))

        # Start polling messages from the queue
        self.frame.after(100, self.poll_log_queue)

    def display(self, record: str) -> None:
        """Passes message to be displayed to the console and scrolls to the bottom"""
        msg = self.queue_handler.format(record)
        self.scrolled_text.configure(state='normal')
        self.scrolled_text.insert(tk.END, f'{msg}\n', record.levelname)
        self.scrolled_text.configure(state='disabled')
        self.scrolled_text.see(tk.END)

    def poll_log_queue(self) -> None:
        """Checks every 100ms if there is a new message in the queue to display"""
        while True:
            try:
                record = self.log_queue.get(block=False)
            except queue.Empty:
                break
            else:
                self.display(record)
        self.frame.after(100, self.poll_log_queue)

#-------------------------------- GUI Classes ---------------------------------#

class RadioButtonRow:
    """Settings Row with a Label and 2 Radio Buttons"""

    def __init__(
            self,
            frame,
            title: str,
            options: 'list[str]',
            settings: dict,
            settings_var: str,
            row: int,
        ) -> None:

        self.settings = settings
        self.settings_var = settings_var

        self.v = tk.StringVar()
        self.v.set(self.settings[self.settings_var])

        tk.Label(frame, text=title).grid(column=0, row=row, sticky='w')
        for col, option in enumerate(options, 2):
            tk.Radiobutton(
                frame,
                text=option,
                variable=self.v,
                value=option,
            ).grid(column=col, row=row, sticky='w')


class BrowseFileRow:
    """Settings Row with a Label, an Entry to display selection and a browse Button"""

    def __init__(
            self,
            frame,
            title: str,
            settings: dict,
            settings_var: str,
            row: int,
            mode: str,
            default_cwd: bool = False,
        ) -> None:

        self.settings = settings
        self.settings_var = settings_var
        self.mode = mode

        self.v = tk.StringVar()
        if default_cwd:
            self.v.set(Path.cwd())
        else:
            self.v.set(self.settings[self.settings_var])

        tk.Label(frame, text=title).grid(column=0, row=row, sticky='w', ipadx=10, pady=2)
        path_display_box = tk.Entry(
            frame,
            textvariable=self.v,
            border=0.5,
            relief='sunken',
        )
        path_display_box.grid(column=1, columnspan=2, row=row, sticky='ew', ipadx=5, ipady=3, pady=2)
        frame.grid_columnconfigure(1, weight=5)
        path_display_box.xview('end')
        tk.Button(
            frame,
            text='Browse...',
            command=self.browse_callback,
        ).grid(column=3, row=row, sticky='e', ipadx=2, padx=(0, 10), pady=2)

    def browse_callback(self) -> None:
        if self.mode == 'dir':
            selection = filedialog.askdirectory(initialdir=Path.cwd())
            if selection != '':
                self.v.set(selection)
        elif self.mode == 'file':
            selection = filedialog.askopenfilename(
                initialdir=Path.cwd(),
                filetypes=[('Hierarchical Data Format (HDF5)', '.h5')]
            )
            if selection != '':
                self.v.set(os.path.basename(selection))


class NumberInputRow:
    """Settings Row with a Label and an Entry for numbers validated with a warning Label"""

    def __init__(
            self,
            frame,
            title: str,
            settings: dict,
            settings_var: str,
            row: int,
            mode: str = '',
        ) -> None:

        self.frame = frame
        self.settings = settings
        self.settings_var = settings_var
        self.row = row
        self.mode = mode

        self.v = tk.IntVar()
        self.v.set(self.settings[self.settings_var])

        reg = self.frame.register(self.input_callback)

        tk.Label(self.frame, text=title).grid(column=0, row=self.row, sticky='w', ipadx=10, pady=3.5)
        tk.Entry(
            self.frame,
            textvariable=self.v,
            border=0.5,
            relief='sunken',
            width=9,
            justify='right',
            validate='all',
            validatecommand=(reg, '%P'),
        ).grid(column=3, row=self.row, sticky='e', ipadx=2, padx=(0, 10), pady=2)

    def input_callback(self, input_var: tk.IntVar) -> bool:
        self.hide_error()

        if str.isdigit(input_var):
            check_input = int(input_var)
            if self.mode == 'not_zero':
                if check_input == 0:
                    self.show_error('Cannot be 0')
            elif self.mode == 'power_of_two':
                if check_input == 0 or (check_input & (check_input-1) != 0):
                    self.show_error('Must be power of 2 (1, 2, 4, 8, 16, 32, etc)')
            elif self.mode == 'percentage':
                if check_input < 0 or check_input > 100:
                    self.show_error('Must be between 0 - 100')
            return True
        elif input_var == '':
            self.show_error('Cannot be empty')
            return True
        else:
            return False

    def show_error(self, error_text: str) -> None:
        self.error_msg = tk.Label(self.frame, text=error_text, fg='red')
        self.error_msg.grid(column=1, columnspan=2, row=self.row, sticky='e')

    def hide_error(self) -> None:
        if hasattr(self, 'error_msg'):
            self.error_msg.grid_forget()


class SettingsUI:
    """UI Section that houses all the settings rows

    Importantly, it also allows users to change the mode to PRODUCTION/TRAINING
    based on the radio button row at the top. This section will redraw itself
    after detecting a change in the model. 

    The current settings obtained from the other rows can also be saved back
    into the settings.yaml file to update users' preferences on subsequent runs. 
    """

    def __init__(self, frame) -> None:
        self.frame = frame
        with open('settings.yaml', 'r') as f: self.settings = yaml.safe_load(f)

        self.adcs_mode = tk.StringVar()
        self.adcs_mode.set(self.settings['adcs_mode'])

        self.draw_adcs_mode_row()
        self.check_adcs_mode(self.settings['adcs_mode'])

    def draw_adcs_mode_row(self) -> None:
        tk.Label(self.frame, text='ADCS Mode').grid(column=0, row=0, sticky='w')

        options = ['PRODUCTION', 'TRAINING']
        for col, option in enumerate(options, 2):
            tk.Radiobutton(
                self.frame,
                text=option,
                variable=self.adcs_mode,
                value=option,
                command=self.redraw_callback,
            ).grid(column=col, row=0, sticky='w')

    def redraw_callback(self) -> None:
        """Clears section before redrawing the rows"""
        for child in self.frame.winfo_children():
            child.destroy()

        self.draw_adcs_mode_row()
        self.check_adcs_mode(self.adcs_mode.get())

    def check_adcs_mode(self, mode: str) -> None:
        if mode == 'PRODUCTION':
            self.draw_production_rows()
        elif mode == 'TRAINING':
            self.draw_training_rows()

    def draw_production_rows(self) -> None:
        browsefile_rows = self.draw_browsefile_rows()
        tk.Label(self.frame, text='Model Configurations', font='TkFixedFont 9 underline').grid(column=0, row=6, sticky='w', pady=(10,0))
        row7  = NumberInputRow(self.frame, 'Batch Size (Higher = more RAM)', self.settings, 'BATCH_SIZE', row=7, mode='power_of_two')
        row8  = NumberInputRow(self.frame, '% Confidence Threshold', self.settings, 'CONF_THRESHOLD', row=8, mode='percentage')
        row9 = BrowseFileRow(self.frame, 'Backside Model', self.settings, 'bs_model', row=9, mode='file')
        row10 = BrowseFileRow(self.frame, 'Edgenormal Model', self.settings, 'en_model', row=10, mode='file')
        self.rows = [*browsefile_rows, row7, row8, row9, row10]

    def draw_training_rows(self) -> None:
        row1 = RadioButtonRow(self.frame, 'Training Mode', ['EDGENORMAL', 'BACKSIDE'], self.settings, 'training_subdir', row=1)
        browsefile_rows = self.draw_browsefile_rows()
        tk.Label(self.frame, text='Training Configurations', font='TkFixedFont 9 underline').grid(column=0, row=6, sticky='w', pady=(10,0))
        row7  = NumberInputRow(self.frame, 'Batch Size (Higher = more RAM)', self.settings, 'BATCH_SIZE', row=7, mode='power_of_two')
        row8  = NumberInputRow(self.frame, 'No. of Models to Train', self.settings, 'training_runs', row=8, mode='not_zero')
        row9 = NumberInputRow(self.frame, 'Balanced no. of Samples per Class', self.settings, 'training_n', row=9, mode='not_zero')
        row10 = NumberInputRow(self.frame, '% Model Saving Threshold', self.settings, 'training_saving_threshold', row=10, mode='percentage')
        self.rows = [row1, *browsefile_rows, row7, row8, row9, row10]

    def draw_browsefile_rows(self) -> 'list[BrowseFileRow]':
        """Static rows for user to select the correct folder locations"""
        tk.Label(self.frame, text='Folder Locations', font='TkFixedFont 9 underline').grid(column=0, row=2, sticky='w', pady=(10,0))
        row3 = BrowseFileRow(self.frame, 'Path to New Images (Please Check!)', self.settings, 'adc_drive_new', row=3, mode='dir')
        row4 = BrowseFileRow(self.frame, 'Path to Old Images (Please Check!)', self.settings, 'adc_drive_old', row=4, mode='dir')
        row5 = BrowseFileRow(self.frame, 'Path to K Drive (Please Check!)', self.settings, 'k_drive', row=5, mode='dir')
        return [row3, row4, row5]

    def load_current_settings(self) -> dict:
        """Gathers all rows' Tkinter variables and store into an updated settings dictionary"""
        current_settings = self.settings.copy()
        current_settings['adcs_mode'] = self.adcs_mode.get()
        for row in self.rows:
            current_settings[row.settings_var] = row.v.get()
        return current_settings

    def save_settings(self) -> None:
        """Gets the updated settings dictionary and save it back into the settings.yaml file"""
        self.settings = self.load_current_settings()
        with open('settings.yaml', 'w') as f:
            yaml.dump(self.settings, f, default_flow_style=False)


class App:
    """Main Tkinter body that packs the SettingsUI, START/STOP and Save buttons, and ConsoleUI"""

    def __init__(self, root: tk.Tk, log_queue: queue.Queue) -> None:
        self.root = root

        # Tkinter Root Window Configuration
        self.root.title('ADC System')
        self.root.wm_iconbitmap('assets/icon.ico')
        self.root.minsize(600, 500)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.protocol('WM_DELETE_WINDOW', self.quit)

        # Stretchable Vertical PanedWindow
        vertical_pane = tk.PanedWindow(self.root, orient='vertical')
        vertical_pane.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

        # Top LabelFrame for Settings UI
        settings_frame = tk.LabelFrame(vertical_pane, text='Settings', padx=5, pady=5)
        settings_frame.columnconfigure(0, weight=1) # makes col 0 (the only col) stretchable
        vertical_pane.add(settings_frame, minsize=345)

        # Middle Frame for red 'Running' Label and START/STOP and Save Buttons
        btns_frame = tk.Frame(vertical_pane)
        self.save_btn = tk.Button(
            btns_frame,
            text='Save',
            command=self.save_callback,
            width=8,
        )
        self.save_btn.pack(side='right', anchor='n', padx=(0, 17), pady=(5, 2))

        self.adcs_started = tk.BooleanVar()
        self.adcs_started.set(False)
        self.start_stop_btn_text = tk.StringVar()
        self.start_stop_btn_text.set('START')
        tk.Button(
            btns_frame,
            textvariable=self.start_stop_btn_text,
            command=self.start_stop_callback,
            width=8,
        ).pack(side='right', anchor='n', padx=(0, 17), pady=(5, 2))

        self.is_running = tk.Label(
            btns_frame,
            text='Running',
            fg='red',
        )
        self.is_running.pack_forget()
        vertical_pane.add(btns_frame, minsize=33)

        # Bottom LabelFrame for Console UI
        console_frame = tk.LabelFrame(vertical_pane, text='Console', padx=5, pady=5)
        console_frame.rowconfigure(1, weight=1) # makes textbox stretchable
        console_frame.columnconfigure(0, weight=1) # makes col 0 (the only col) stretchable
        vertical_pane.add(console_frame)

        # Initialize all frames
        self.settings_ui = SettingsUI(settings_frame)
        self.console_ui = ConsoleUI(console_frame, log_queue)

    def quit(self, *args) -> None:
        logging.info('Safely shutting down ADCS...')
        self.root.destroy()

    def save_callback(self) -> None:
        self.settings_ui.save_settings()
        logging.info('SETTINGS SAVED')

    def start_stop_callback(self) -> None:
        if not self.adcs_started.get(): # was stopped >> to start
            if not (hasattr(self, 'adcs_thread') and self.adcs_thread.is_alive()):

                # Disable all Tkinter Radio Buttons/Buttons/Entry Boxes once started
                self.user_inputs = [*self.settings_ui.frame.winfo_children(), self.save_btn]
                for child in self.user_inputs:
                    if isinstance(child, tk.Radiobutton) or isinstance(child, tk.Button) or isinstance(child, tk.Entry):
                        child.configure(state='disabled')
                self.is_running.pack(side='right', anchor='n', padx=(0, 17), pady=(6, 2))

                # Start either Production/Training thread
                if self.settings_ui.adcs_mode.get() == 'PRODUCTION':
                    self.adcs_thread = Production(
                        self.settings_ui.load_current_settings(),
                        self.adcs_started,
                        self.start_stop_btn_text,
                        self.is_running,
                        self.user_inputs,
                    )
                elif self.settings_ui.adcs_mode.get() == 'TRAINING':
                    self.adcs_thread = Training(
                        self.settings_ui.load_current_settings(),
                        self.adcs_started,
                        self.start_stop_btn_text,
                        self.is_running,
                        self.user_inputs,
                    )

            self.adcs_started.set(True)
            self.start_stop_btn_text.set('STOP')
            self.adcs_thread.start()
        else: # was started >> to stop
            self.adcs_thread.stop()

#--------------------------------- Start GUI ----------------------------------#

def main() -> None:
    root = tk.Tk()
    log_queue = queue.Queue()
    logging.basicConfig(
        format='[%(levelname)s] %(asctime)s - %(message)s',
        datefmt='%d-%b-%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                filename='debug.log',
                mode='w',
            ),
            QueueHandler(log_queue),
            # logging.StreamHandler(), # for logging to python console
        ],
    )

    try:
        app = App(root, log_queue)
        logging.info('[Automatic Defect Classification System]')
        logging.info('// By: Tam Zher Min')
        logging.info('// User Guide: https://github.com/zhermin/ADCS')
        logging.info('Click START to begin selected mode')
        app.root.mainloop()
    except KeyboardInterrupt:
        sys.exit('\nSafely shutting down ADCS...')
    except Exception as e:
        logging.exception('ERROR OCCURRED')

if __name__ == '__main__':
    main()
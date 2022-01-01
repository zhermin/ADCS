import os, time, shutil
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

from dataclasses import dataclass

@dataclass
class Wafer:
    """Simple, fixed data structure to store each Wafer information

    # Description
    This is a dataclass which is similar to a dictionary but its
    intention is clearer. Slots help save a bit of memory by
    locking the attributes but no new attributes can be created
    at runtime. 

    # Usage Notes
    Access is through dot notation (wafer.data) instead of
    square-bracket notation (wafer['data']). 
    """

    __slots__ = (
        'filename',
        'classnumber_row',
        'classnumber_col',
        'classnumber',
        'data',
        'exists',
    )

    filename: str # Exact filename in KLA file after the word TiffFileName
    classnumber_row: int # Row (line number) that the CLASSNUMBER is found
    classnumber_col: int # Column that the CLASSNUMBER is found
    classnumber: str # The CLASSNUMBER in string format, not integer!
    data: list # The entire classnumber_row split into a list by space
    exists: bool # Whether this file can be found in the ADCS' "new" folder


class KlaReader:
    """Parses KLA file and sorts each wafer filename according to their CLASSNUMBER

    # Description
    This reader is bug-prone in that the way the relevant data are extracted is to
    use some line and string splitting and formatting. Hence, this reader is operating
    under the major assumption that the formats of KLA files will be the same. 

    In my own testing, I have found that KLA files can be in 2 formats and I have
    accounted for them using an 'offset' variable to get the correct line. However,
    I cannot guarantee that there won't be more formats. So if for some wafer lots,
    the wafers are sorted wrongly or can't be found even if they are in the folder, 
    then this parser might need to be tweaked for those cases. 

    Nonetheless, this works for all the KLA files I have so far. After parsing them, 
    it also allows easy edits to the KLA file by editing the wafer.data attribute for
    each wafer then editing the particular line number using the wafer.classnumber_row
    attribute. At the end, we can just combine all edited lines to make the KLA file. 

    # Usage Notes
    This reader can also be used to print out information from a particular KLA file if
    run directly, which you can try and understand from the few lines at the bottom. 
    You can also find files containing a specified CLASSNUMBER using the 
    get_defects_by_classnumber() function. Not too useful in production mode but just FYI. 
    """

    def __init__(self, kla_file: Path) -> None:
        """Inits KlaReader, opens kla_file and runs functions to sort defects"""
        self.kla_file = kla_file

        with open(self.kla_file, mode='r', encoding='utf-8') as f:
            self.lines = f.readlines()

        self.read_kla()
        self.sort_defects()

    def read_kla(self) -> None:
        """Parses KLA file and creates a Wafer dataclass for each filename found

        Firstly, we try to find the word 'CLASSNUMBER' in each line of the file.
        If we found it, we know that we have reached the parts where all the filenames
        and defect information are stored. Then, we get the correct row and column
        numbers using some offset and splitting logic before storing that line as well.
        At the end, all information in each wafer will be stored in the self.wafers list. 
        """

        # Default index: 10th column = index 9 (index starts from 0)
        # This is the index that the word 'CLASSNUMBER' is always found
        classnumber_index = 9

        # By using a simple 'offset' variable, we account for 2 KLA file formats
        # This is because it is observed that the lines where the information are
        # stored in either formats can be obtained by either adding 0 or 1. 
        # You need to read the KLA file yourself to fully understand the logic. 
        if ''.join(self.lines).count('DefectRecordSpec') == 1: # Format 1: 'FS'
            offset = 0
        else: # Format 2: 'BS/EN'
            offset = 1

        self.wafers = []
        for i in range(len(self.lines)):
            split_line = self.lines[i].split(' ')

            # Each set of lines with the word 'CLASSNUMBER' contain wafer information
            if 'CLASSNUMBER' in split_line:
                classnumber_index = split_line.index('CLASSNUMBER') - 2

            # The filename is after the word 'TiffFileName'
            if 'TiffFileName' in split_line:
                filename = split_line[1].strip().replace(';','')
                classnumber_row = i+2+offset
                classnumber_col = classnumber_index+offset

                data = self.lines[classnumber_row].split(' ')

                # All wafer maps seem to have '2000.000' in their data (bug-prone)
                if '2000.000' in data:
                    # Change wafer maps' code from 174 (Backside) to 
                    # 172 (Probemark Shift) to avoid predicting on them. 
                    classnumber = '172'
                else:
                    classnumber = data[classnumber_col]

                new_wafer = Wafer(filename, classnumber_row, classnumber_col, classnumber, data, False)
                self.wafers.append(new_wafer)

    def get_defects_by_classnumber(self, classnumber: str) -> 'list[Wafer]':
        """Returns a list of Wafers with a specified classnumber"""
        return [wafer for wafer in self.wafers if wafer.classnumber == str(classnumber)]

    def sort_defects(self) -> None:
        """Sorts each wafer into separate lists

        Based on the wafer's classnumber, we can sort them into lists so that
        our models can know which wafers to work on. For example, a backside model
        should only work on self.bs_wafers. 

        For wafer classnumbers that are not in these conditionals, we consider them as
        unclassified. These include code 175 (Edge Segmentation), etc. If there are 
        unclassified wafers found, the ADCS will prompt in the logger so that maybe 
        further actions can be done to sort them correctly or create new conditionals. 
        """

        self.fs_wafers, self.bs_wafers, self.en_wafers, self.et_wafers, self.wafer_maps, self.unclassified = [], [], [], [], [], []
        for wafer in self.wafers:
            if wafer.classnumber == '56':
                # "[056] AVI Def"
                self.fs_wafers.append(wafer)
            elif wafer.classnumber == '174':
                # "[174] AVI_Backside Defect"
                self.bs_wafers.append(wafer)
            elif wafer.classnumber == '173':
                # "[173] AVI_Bevel Defect"
                self.en_wafers.append(wafer)
            elif wafer.classnumber == '176':
                # "[176] AVI_Edge Top Defect"
                self.et_wafers.append(wafer)
            elif wafer.classnumber == '172':
                # "[172] AVI_Probemark Shift"
                self.wafer_maps.append(wafer)
            else:
                # CLASSNUMBERS that don't belong in above cases will be ignored
                self.unclassified.append(wafer)

        # Send warning in the logger so that operator can look into these unclassified classnumbers
        if len(self.unclassified) > 0:
            unclassified_classnumbers = set([int(wafer.classnumber) for wafer in self.unclassified])
            logging.warning(f'Ignoring unsorted CLASSNUMBERS ({", ".join([f"#{num}" for num in unclassified_classnumbers])})')

    def check_imgs_in_kla(
            self,
            directory: Path,
            retry_num: int = 5,
            timeout: int = 10,
        ) -> None:
        """Check if filenames mentioned in KLA file can be found in the folder

        This checks the specified directory to see if the wafers exist in it or not. 
        If it does, set wafer.exists = True, else wafer.exists = False. This helps us 
        skip wafers that can't be found so the ADCS doesn't keep trying to find them.

        This function will also try as many times as retry_num and wait timeout
        seconds before retrying just in case the AXI machine is still loading the 
        images into the folder. Set these to higher values to ensure all images 
        are accounted for before classification but at the cost of a slower system. 
        """

        for tries in range(retry_num,0,-1):
            folder_imgs = []
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff']
            for ext in extensions:
                folder_imgs.extend(directory.glob(ext))

            folder_imgs = [folder_img.name for folder_img in folder_imgs]

            for wafer in self.wafers:
                if wafer.filename in folder_imgs:
                    wafer.exists = True
                else:
                    wafer.exists = False

            wafers_not_found = [int(wafer.classnumber) for wafer in self.wafers if not wafer.exists]
            if len(wafers_not_found) > 0:
                logging.warning(f'{len(wafers_not_found)} images missing ({", ".join([f"#{num}" for num in set(wafers_not_found)])}) (Retry = {tries})')
                time.sleep(timeout)
            else:
                break

        if len(wafers_not_found) > 0:
            logging.warning(f'Ignoring the missing {len(wafers_not_found)} images referenced')
        else:
            logging.info('Succesfully found all images')

    def edit_kla(self, wafers: 'list[Wafer]', prediction: 'list[int]') -> 'list[Wafer]':
        """Edits KLA file with predicted classnumbers

        For each wafer and its corresponding predicted class, 
        we edit the data attribute at the classnumber_col index
        to the new classnumber. Then we edit the line at the 
        classnumber_row index by joining the data line. At the end, 
        we merge everything together and overwrite the KLA file. 
        We also return the wafers at the end so that they are updated. 

        Args:
            wafers (list of Wafer): The list of wafers to be edited
            prediction (list of int): 
                The prediction column from the predictor dataframe
                that has the correct classnumbers. 

        Returns:
            wafers (list of Wafer): The input list of wafers, updated
        """

        for wafer, pred in zip(wafers, prediction):
            wafer.classnumber = str(pred)
            wafer.data[wafer.classnumber_col] = wafer.classnumber
            self.lines[wafer.classnumber_row] = ' '.join(wafer.data)

        with open(self.kla_file, mode='w', encoding='utf8') as f: 
            f.writelines(self.lines)

        return wafers

    def move_predicted_lots(
            self,
            wafers: 'list[Wafer]',
            source: Path,
            copy_destination: Path,
            move_destination: Path,
            defect_mapping: 'dict[str, int]' = None,
        ) -> 'list[Wafer]':
        """Moves and copies the predicted wafer lots

        There are 2 folders taken into account: K drive, where KLARITY Defect
        reads all the files; and adcs_drive old, where we store the backups 
        in the 'unsorted' subfolder. 

        Firstly, we copy the KLA file to those 2 destinations. Then for each
        wafer, if it exists, we move them to the correct subfolders according
        to their predicted classnumber. For example, if a wafer was predicted 
        to be chipping, we move it to the old folder's chipping subfolder. 

        All other wafers are moved to the unclassified folder and into their
        respective subfolder based on their original classnumber. 

        This allows operators to easily sort through the unsorted folder and 
        move them to the correct ones in trainval so that retraining is accurate. 
        We also return the wafers at the end so that they are updated. 

        Args:
            wafers (list of Wafer): The list of wafers to be moved
            source (Path): The source path where the files are found, adcs_drive_new
            copy_destination (Path): The path to be copied to, K drive
            move_destination (Path): 
                The path to be moved to, adcs_drive_old and its correct subdir,
                eg. either in backside or edgenormal subdir. 
            defect_mapping (dict{str: int}), default = None:
                The correct defect_mapping, either bs_ or en_.
                Set from settings.yaml file. 

        Returns:
            wafers (list of Wafer): The input list of wafers, updated
        """

        # KLA File
        shutil.copy2(str(Path.joinpath(source, self.kla_file)), str(copy_destination))
        shutil.copy2(str(Path.joinpath(source, self.kla_file)), str(move_destination))
        moved_imgs = 1

        # All Wafer Scans
        for wafer in wafers:
            if not wafer.exists:
                continue

            file_to_move = Path.joinpath(source, wafer.filename)

            # CLASSNUMBERS in brackets below will NOT be sent to K Drive
            # but will still be stored into the "old" folder for backup.
            if wafer.classnumber not in ('0', '176'):
                shutil.copy2(str(file_to_move), str(copy_destination))

            wafer.exists = False
            moved_imgs += 1

            if defect_mapping is not None:
                defect_mapping_inv = {v: k for k, v in defect_mapping.items()}
                defect_folder = Path.joinpath(move_destination, defect_mapping_inv.get(int(wafer.classnumber)))
            else:
                if wafer.classnumber == '56': # Frontside images will be in its own folder
                    defect_folder = move_destination
                else:
                    defect_folder = Path.joinpath(move_destination, wafer.classnumber)

            defect_folder.mkdir(parents=True, exist_ok=True)
            shutil.move(str(file_to_move), str(Path.joinpath(defect_folder, wafer.filename)))

        logging.info(f'{moved_imgs} file(s) copied and moved')
        return wafers

    def remove_kla_file(self) -> None:
        """Removes the original KLA file after all images have been copied and moved"""
        os.remove(self.kla_file)

    def __str__(self) -> str:
        """String representation for testing purposes when printing this KlaReader object"""
        return ''.join([
            f'({i:03}) {wafer.filename} :: {wafer.classnumber} :: {"Found" if wafer.exists else ""}\n'
            for i, wafer in enumerate(self.wafers, 1)
        ])


def main() -> None:
    """Reading a command-line argument specified KLA file

    You can run this script directly by specifying
    the full path of a KLA file you want to read. 

    Usage Notes
    CMD >> python {path_to_script/kla_reader.py} BESUS542.1-CARRIER_A1_01.000
    """

    import sys
    path = Path(sys.argv[1])
    reader = KlaReader(path)
    reader.check_imgs_in_kla(directory=path.parent, retry_num=1, timeout=0)
    print(reader)

if __name__ == '__main__':
    main()
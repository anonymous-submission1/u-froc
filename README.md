# u-froc

Hello! :vulcan_salute:

Here you may find a plenty of useful and lots of useless code.

### Setting up the Libraries

###### 1. Install `u-froc` module:

```
git clone git@github.com:anonymous-submission1/u-froc.git
cd u-froc
pip install -e .
```
### Data downloading and preprocessing

For all datasets you need to specify the local path 
to a folder with the preprocessed dataset in the file `ufroc/paths.py`.

#### LIDC dataset
Download the dataset
[here](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI).
For the benchmarking purposes you need to download only "Images (DICOM, 125GB)".
Firstly, download its `.tcia` file.

Then we suggest to use NBIA Data Retriever CLI.
We also use its open source version so that NBIA Data Retriever
cloud be installed without the super user permission.

If you prefer to use NBIA Data Retriever GUI,
feel free to skip the steps below and follow the TCIA guidelines.

1. Download the `.tcia` file and place it anywhere on your computer,
e.g., `~/TCIA_LIDC-IDRI_20200921.tcia`.

2. Clone
[NBIA Data Retriever CLI repo](https://github.com/ygidtu/NBIA_data_retriever_CLI):
`git clone git@github.com:ygidtu/NBIA_data_retriever_CLI.git`

3. Install go in your conda.
(Go is needed to build the data retriever scripts.)
`conda install -c conda-forge go`

4. Build the data receiver scripts:
(1) `cd NBIA_data_retriever_CLI`, (2) `chmod +x build.sh`, (3) `./build.sh`.
    
5. Run the downloading script:

```
./nbia_cli_linux_amd64 -i ~/TCIA_LIDC-IDRI_20200921.tcia -o <raw_data_path>
``` 


You need to specify `<raw_data_path>`, where to download the data.
You may also have a platform different from linux
so you need to choose another available downloading script.   

Finally, run our preprocessing script:

```
python scripts/preproc_lidc.py -i <raw_data_path>/LIDC-IDRI -o <preprocess_data_path>
```

`raw_data_path` should be a path to a folder `LIDC-IDRI`,
which contains folders `LIDC-IDRI-{i}`.

Now you can use `<preprocess_data_path>`
as a path to the LIDC dataset in `ood/paths.py`


#### MIDRC dataset

MIDRC dataset can be downloaded on
[TCIA web page](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=80969742).
For the benchmarking purposes you need to download "Images (DICOM, 11 GB)"
and "Annotations (JSON)" files.

The steps are the same as in LIDC downloading and preprocessing
except the file names.
So here we skip NBIA Data Retriever CLI installation steps;
see [LIDC dataset section](#LIDC dataset).

1. Download the `.tcia` file and place it anywhere on your computer,
e.g., `~/MIDRC-RICORD-1a.tcia`.

2. Run the downloading script:
`~/NBIA_data_retriever_CLI/nbia_cli_linux_amd64 -i ~/MIDRC-RICORD-1a.tcia -o <raw_data_path>`. 
You need to specify `<raw_data_path>`, where to download the data.

3. Download `.json` annotations file, unzip it, and place in folder `<raw_data_path>`.
(Keep its filename the same.)

4. Run our preprocessing script:
`python scripts/preproc_midrc.py -i <raw_data_path> -o <preprocess_data_path>`

Now you can use `<preprocess_data_path>`
as a path to the MIDRC dataset in `ood/paths.py`


#### LiTS dataset

For the LiTS dataset you need to download LiTS competition
[training data](https://competitions.codalab.org/competitions/17094).

*****

*Private shortcut to download the data:*


1. `pip install gdown`
2. `cd <raw_data_path>`
3. `gdown "https://drive.google.com/uc?id=0B0vscETPGI1-TE5KWFgxaURubFE&resourcekey=0-0fwNqxVQJSfYDvSt1Kr_Sg"`
4. `gdown gdown "https://drive.google.com/uc?id=0B0vscETPGI1-cTZGbTU4UC05Qm8&resourcekey=0-qf26pTzXmgVv_qznEBDNqQ"`

*****

Preprocessing instructions:

1. Download both archives (`Training_Batch1.zip` and `Training_Batch2.zip`) in folder
`raw_data_path`. Then unzip them there: 
    - `cd <raw_data_path>`
    - `unzip Training_Batch1.zip`
    - `unzip Training_Batch2.zip`
    
2. Run our preprocessing script:
`python scripts/preproc_lits.py -i <raw_data_path> -o <preprocess_data_path>`

Now you can use `<preprocess_data_path>`
as a path to the LiTS dataset in `ood/paths.py`

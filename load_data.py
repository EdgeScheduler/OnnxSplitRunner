from tvm import relay
import onnx
import os
from config import Config
from Onnxs.download_config import OnnxModelUrl
import time
import requests 
from tqdm import tqdm

def easy_load_from_onnx(save_name,input_dict={}, download_url=None, auto_path=True, validate_download=False):
    '''
    load model from "$project_path/onnxs/$save_name/$save_name.onnx" or "$save_name" without any redundant operate. It may not fit some complex model.

    Parameters
    ----------
    save_name : str
        onnx-file name. if auto_path is True, load file from "$project_path/onnxs/$save_name/$save_name.onnx", or load from "$save_name" directly.
    input_dict : dict => {str: tuple}
        input-label to tensor-shape
    download_url : str
        url to download onnx-model from internet, it only works when local-file is not exist.
    auto_path: bool
        transform "$save_name" to "$project_path/onnxs/$save_name/$save_name.onnx"
    validate_download: bool
        check if file is complete. if not, download the left.

    Returns
    -------
    irModule : tvm.ir.module.IRModule
        The relay module for compilation

    params : dict => {"label": tvm.nd.NDArray}
        The parameter dict to be used by relay
    load_time: float
        how long to load onnx
    '''

    start=time.time()
    filepath=save_name
    if auto_path:
        filepath=Config.ModelSavePathName(filepath)

    if not os.path.exists(filepath):
        if len(download_url)<1 or download_url is None:
            print("onnx file not exist, you can give download url by set download_url=$URL.")
            return None,{},time.time()-start
        else:
            if not download(download_url,filepath):
                print("fail to download onnx-model to:",filepath)
                return None,{},time.time()-start
    else:
        if validate_download:
            if not download(download_url,filepath):
                print("fail to validate and download onnx-model to:",filepath)
                return None,{},time.time()-start

    try:
        onnx_model=onnx.load(filepath)
        irModule, params = relay.frontend.from_onnx(onnx_model,input_dict)
        irModule=relay.transform.InferType()(irModule)                    # tvm.ir.module.IRModule
    except Exception as ex:
        print("fail to load onnx from %s, error info: %s"%(filepath,str(ex)))
        return None,{},time.time()-start

    print("success to load onnx from %s"%(filepath))
    return irModule, params,time.time()-start
    
def download(url,path)->bool:
    '''
    Downloads the file from the internet, resumable data is supported. Set the input options correctly to overwrite or do the size comparison

    Parameters
    ----------
    url : str
        Download url.

    path : str
        Local file path to save downloaded file.

    retries: int, optional
        Number of time to retry download, defaults to 3.

    Returns
    -------
    result : bool
        success or fail
    '''

    print("start to download file from:",url)

    start=time.time()
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)),exist_ok=True)
        Response = requests.get(url, stream=True)
        total_size = int(Response.headers.get('content-length', 0))
        
        # judge if file already in disk and read temp-size
        if os.path.exists(path):
            temp_size = os.path.getsize(path)
        else:
            temp_size = 0

        if temp_size >= total_size:
            print("complete file already exists, skip the download.")
            return True
        
        # start to download left-data
        Req = requests.get(url,headers={"Range": f"bytes={temp_size}-{total_size}"},stream=True)
        with open(path, "ab") as file, tqdm(initial=temp_size,desc="downloading to "+path,total=total_size,unit='iB',unit_scale=True,unit_divisor=1024,) as bar:
            for data in Req.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    except Exception as ex:
        print("fail to download from %s, cost time=%f min. error info: %s"%(url,(time.time()-start)/60.0,str(ex)))
        return False

    print("success download from %s and save to %s, cost time=%f min."%(url,path,(time.time()-start)/60.0))
    return True

# def download(url, path)->bool:
#     '''
#     Downloads the file from the internet. Set the input options correctly to overwrite or do the size comparison

#     Parameters
#     ----------
#     url : str
#         Download url.

#     path : str
#         Local file path to save downloaded file.

#     retries: int, optional
#         Number of time to retry download, defaults to 3.

#     Returns
#     -------
#     result : bool
#         success or fail
#     '''
    
#     print("start to download file from:",url)
#     start=time.time()

#     try:
#         with open(path, "wb") as code:
#             code.write(requests.get(url).content)
#     except Exception as ex:
#         print("fail to download from %s, cost time=%fs. error info: %s"%(url,time.time()-start,str(ex)))
#         return False

#     print("success download from %s and save to %s, cost time=%fs."%(url,path,time.time()-start))
#     return True

# def download(url: str, path: str)->bool:
#     '''
#     Downloads the file from the internet. Set the input options correctly to overwrite or do the size comparison

#     Parameters
#     ----------
#     url : str
#         Download url.

#     path : str
#         Local file path to save downloaded file.

#     retries: int, optional
#         Number of time to retry download, defaults to 3.

#     Returns
#     -------
#     result : bool
#         success or fail
#     '''
#     print("start to download file from:",url)

#     start=time.time()

#     try:
#         os.makedirs(os.path.dirname(os.path.abspath(path)),exist_ok=True)

#         resp = requests.get(url, stream=True)
#         total = int(resp.headers.get('content-length', 0))
#         with open(path, 'wb') as file, tqdm(desc="downloading to"+path,total=total,unit='iB',unit_scale=True,unit_divisor=1024,) as bar:
#             for data in resp.iter_content(chunk_size=1024):
#                 size = file.write(data)
#                 bar.update(size)
#     except Exception as ex:
#         print("fail to download from %s, cost time=%f min. error info: %s"%(url,(time.time()-start)/60.0,str(ex)))
#         return False

#     print("success download from %s and save to %s, cost time=%f min."%(url,path,(time.time()-start)/60.0))
#     return True
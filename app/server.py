import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse

from starlette.staticfiles import StaticFiles
import numpy as np

export_file_url = 'https://drive.google.com/uc?export=download&id=1XAc-JRqCK2x6jGb27rdJMwwdoEvgcaCe'
export_file_name = 'export.pkl'

classes = ['air_jordan_4', 'air_jordan_1', 'adidas_superstar', 'air_jordan_3', 'air_jordan_2', 'air_jordan_5', 'air_jordan_6', 'air_zoom_pegasus_35', 'asics_gel_contend_4', 'brooks_cascadia_13', 'converse_chuck_taylor_high', 'Lebron_13', 'Lebron_14', 'Lebron_15', 'Lebron_16', 'vans_old_skool']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    
    prediction = learn.predict(img)[0]
    temp = learn.predict(img)[2]
    idx = np.argmax(temp)
    acc = temp[idx]
    
    value = {'Lebron_13': (100, 220), 'Lebron_14': (100, 280), 'Lebron_15': (120, 380), 'Lebron_16': (120, 300), 'adidas_superstar': (80, 150), 'air_jordan_1': (120, 500), 'air_jordan_2': (120, 400), 'air_jordan_3': (120, 420), 'air_jordan_4': (120, 500), 'air_jordan_5': (120, 450), 'air_jordan_6': (120, 320), 'air_zoom_pegasus_35': (50, 120), 'asics_gel_contend_4': (35, 59), 'brooks_cascadia_13': (85, 130), 'converse_chuck_taylor_high': (55, 200), 'vans_old_skool': (60, 200)}
    
    if acc>0.5:
        message = '%s (probability %.02f), current market value is %.02f-%.02f USD.' % (prediction, acc, value[str(prediction)][0], value[str(prediction)][1])
    else:
        message = 'No shoe found, please send us a request to add it to the database.'
    return JSONResponse({'result': str(message)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")

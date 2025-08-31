from .aloha_joycon_handler import AlohaJoyconHandler

class TransferCubeJoyconHandler(AlohaJoyconHandler):
    _name = "transfer_cube_joycon"

    def __init__(self):
        super().__init__()

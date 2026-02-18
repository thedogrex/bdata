import logging
import app.config as config
from py_clob_client.client import ClobClient

class Core:
    def __init__(self):
        self.TAG = self.__class__.__name__

        self.host: str = "https://clob.polymarket.com"
        self.chain_id: int = 137

        print(config.API_KEY)
        print(config.WALLET_ADDRESS)


        self.client = ClobClient(self.host,
                                 key=config.API_KEY,
                                 chain_id=self.chain_id,
                                 signature_type=1,
                                 funder=config.WALLET_ADDRESS)


        print(f'init Core')

        print(self.client.derive_api_key())
        #self._db: DbProvider = DbProvider()


    # ----------------------------------------------------------------------------------
    async def run(self):
        print(f'[{self.TAG}] Run Core!')

        #task_matchmaker = asyncio.create_task(self._match_maker.run())
        #task_matches = asyncio.create_task(self._matches.run())

        #await task_matchmaker
        #await task_matches


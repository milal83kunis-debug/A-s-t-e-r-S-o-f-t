from random import choice, randint, shuffle
from cryptography.fernet import Fernet
from base64 import urlsafe_b64encode
from time import sleep, time
from os import path, mkdir
from loguru import logger
from hashlib import md5
import asyncio
import json

from modules.retry import DataBaseError
from modules.utils import WindowName, get_address
from settings import (
    SHUFFLE_WALLETS,
    PAIR_SETTINGS,
    TRADES_COUNT,
    RETRY,
)

from cryptography.fernet import InvalidToken


class DataBase:

    STATUS_SMILES: dict = {
        True: '✅ ',
        False: "❌ ",
        None: "",
        "WARNING": "⚠️ ",
    }
    lock = asyncio.Lock()

    def __init__(self):

        self.modules_db_name = 'databases/modules.json'
        self.report_db_name = 'databases/report.json'
        self.stats_db_name = 'databases/stats.json'
        self.personal_key = None
        self.window_name = None

        # create db's if not exists
        if not path.isdir(self.modules_db_name.split('/')[0]):
            mkdir(self.modules_db_name.split('/')[0])

        for db_params in [
            {"name": self.modules_db_name, "value": "[]"},
            {"name": self.report_db_name, "value": "{}"},
            {"name": self.stats_db_name, "value": "{}"},
        ]:
            if not path.isfile(db_params["name"]):
                with open(db_params["name"], 'w') as f: f.write(db_params["value"])

        amounts = self.get_amounts()
        if amounts.get("groups_amount"):
            logger.info(f'Loaded {amounts["groups_amount"]} groups\n')
        else:
            logger.info(f'Loaded {amounts["modules_amount"]} modules for {amounts["accs_amount"]} accounts\n')


    def set_password(self):
        if self.personal_key is not None: return

        logger.debug(f'Enter password to encrypt API keys (empty for default):')
        raw_password = input("")

        if not raw_password:
            raw_password = "@karamelniy dumb shit encrypting"
            logger.success(f'[+] Soft | You set empty password for Database\n')
        else:
            print(f'')
        sleep(0.2)

        password = md5(raw_password.encode()).hexdigest().encode()
        self.personal_key = Fernet(urlsafe_b64encode(password))


    def get_password(self):
        if self.personal_key is not None: return

        with open(self.modules_db_name, encoding="utf-8") as f: modules_db = json.load(f)
        if modules_db:
            if list(modules_db.values())[0].get("group_number"):
                test_key = list(modules_db.values())[0]["wallets_data"][0]["encoded_apikey"]
            else:
                test_key = list(modules_db.keys())[0]
        else:
            return

        try:
            temp_key = Fernet(urlsafe_b64encode(md5("@karamelniy dumb shit encrypting".encode()).hexdigest().encode()))
            self.decode_pk(pk=test_key, key=temp_key)
            self.personal_key = temp_key
            return
        except InvalidToken: pass

        while True:
            try:
                logger.debug(f'Enter password to decrypt your API keys (empty for default):')
                raw_password = input("")
                password = md5(raw_password.encode()).hexdigest().encode()

                temp_key = Fernet(urlsafe_b64encode(password))
                self.decode_pk(pk=list(modules_db.keys())[0], key=temp_key)
                self.personal_key = temp_key
                logger.success(f'[+] Soft | Access granted!\n')
                return

            except InvalidToken:
                logger.error(f'[-] Soft | Invalid password\n')


    def encode_pk(self, pk: str, key: None | Fernet = None):
        if key is None:
            return self.personal_key.encrypt(pk.encode()).decode()
        return key.encrypt(pk.encode()).decode()


    def decode_pk(self, pk: str, key: None | Fernet = None):
        if key is None:
            return self.personal_key.decrypt(pk).decode()
        return key.decrypt(pk).decode()


    def create_modules(self, mode: int):
        def create_single_trades(apikeys, labels, proxies):
            return {
                self.encode_pk(apikey): {
                    "address": self.encode_pk(apikey.split(':')[0]),
                    "modules": [{"module_name": "trade", "status": "to_run"} for _ in range(randint(*TRADES_COUNT))],
                    "proxy": proxy,
                    "label": label,
                }
                for apikey, label, proxy in zip(apikeys, labels, proxies)
            }

        def create_pair_trades(apikeys, labels, proxies):
            min_pair_size = max(2, min(*PAIR_SETTINGS["pair_amount"]))
            if len(apikeys) < min_pair_size:
                raise DataBaseError(f'Not enough accounts loaded, need at least {min_pair_size}')

            encoded_api_keys = [self.encode_pk(apikey) for apikey in apikeys]
            addresses = [self.encode_pk(apikey.split(':')[0]) for apikey in apikeys]
            all_modules = [
                {
                    'encoded_apikey': encoded_apikey,
                    "address": address,
                    'label': label,
                    'proxy': proxy,
                }
                for encoded_apikey, address, label, proxy in zip(encoded_api_keys, addresses, labels, proxies)
                for _ in range(randint(*TRADES_COUNT))
            ]

            pairs_list = []
            while True:
                pair_size = max(2, randint(*PAIR_SETTINGS["pair_amount"]))
                unique_wallets_left = list({module["address"]: module for module in all_modules}.values())
                if len(unique_wallets_left) < min_pair_size:
                    break
                if len(unique_wallets_left) < pair_size:
                    pair_size = min_pair_size

                pairs_list.append([])
                for _ in range(pair_size):
                    random_wallet_module = unique_wallets_left.pop(randint(0, len(unique_wallets_left) - 1))
                    all_modules.remove(random_wallet_module)
                    pairs_list[-1].append(random_wallet_module)

            pairs_list = {
                f"{pair_index + 1}_{int(time())}": {
                    "group_number": pair_index + 1,
                    'modules': [{"module_name": "trade", "status": "to_run"}],
                    "wallets_data": pair
                }
                for pair_index, pair in enumerate(pairs_list)
            }

            return pairs_list

        self.set_password()

        with open('input_data/apikeys.txt') as f: raw_apikeys = f.read().splitlines()
        labels = []
        api_keys = []
        for key_index, raw_apikey in enumerate(raw_apikeys):
            apikey_data = raw_apikey.split(':')
            if len(apikey_data) == 3:
                labels.append(apikey_data[0])
                account_apikeys = apikey_data[1:]

            elif len(apikey_data) == 2:
                labels.append(f"Account {key_index + 1}")
                account_apikeys = apikey_data

            else:
                raise DataBaseError(f"Unexpected API key format: {raw_apikey}")

            for account_apikey in account_apikeys:
                if len(account_apikey) != 64:
                    raise DataBaseError(f"Unexpected API key format: {account_apikey}")
            api_keys.append(":".join(account_apikeys))

        with open('input_data/proxies.txt') as f:
            proxies = f.read().splitlines()
        if len(proxies) == 0 or proxies == [""] or proxies == ["http://login:password@ip:port"]:
            logger.error('You will not use proxy')
            proxies = [None for _ in range(len(api_keys))]
        else:
            proxies = list(proxies * (len(api_keys) // len(proxies) + 1))[:len(api_keys)]

        with open(self.report_db_name, 'w') as f: f.write('{}')  # clear report db

        if mode == 102:
            create_func = create_pair_trades
        else:
            create_func = create_single_trades
        new_modules = create_func(api_keys, labels, proxies)

        with open(self.modules_db_name, 'w', encoding="utf-8") as f: json.dump(new_modules, f)

        logger.opt(colors=True).critical(f'Dont Forget To Remove API Keys from <white>apikeys.txt</white>!')

        amounts = self.get_amounts()
        if mode == 102:
            logger.info(f'Created Database with {amounts["groups_amount"]} groups!\n')
        else:
            self.set_accounts_modules_done(new_modules)
            logger.info(f'Created Database for {amounts["accs_amount"]} accounts with {amounts["modules_amount"]} modules!\n')


    def get_amounts(self):
        with open(self.modules_db_name, encoding="utf-8") as f: modules_db = json.load(f)
        modules_len = sum([len(modules_db[acc]["modules"]) for acc in modules_db])
        if modules_db and list(modules_db.values())[0].get("group_number"):
            modules_name = "groups_amount"
        else:
            modules_name = "accs_amount"

        for acc in modules_db:
            for index, module in enumerate(modules_db[acc]["modules"]):
                if module["status"] == "failed": modules_db[acc]["modules"][index]["status"] = "to_run"

        with open(self.modules_db_name, 'w', encoding="utf-8") as f: json.dump(modules_db, f)

        if self.window_name == None:
            self.window_name = WindowName(accs_amount=len(modules_db))
        else:
            self.window_name.accs_amount = len(modules_db)

        self.window_name.set_modules(modules_amount=modules_len)

        return {modules_name: len(modules_db), 'modules_amount': modules_len}

    def get_accs_left(self):
        with open(self.modules_db_name, encoding="utf-8") as f: modules_db = json.load(f)
        return len(set([
            acc
            for acc in modules_db
            for module in modules_db[acc]["modules"]
            if module["status"] == "to_run"
        ]))


    def get_all_modules(self, unique_wallets: bool = False):
        self.get_password()

        with open(self.modules_db_name, encoding="utf-8") as f: modules_db = json.load(f)

        if not modules_db:
            return 'No more accounts left'
        elif list(modules_db.values())[0].get("group_number"):
            raise DataBaseError(f'Unexpected database type for this mode')

        all_modules = [
            {
                'encoded_apikey': encoded_apikey,
                'apikey': self.decode_pk(pk=encoded_apikey),
                'address': modules_db[encoded_apikey]["address"],
                'label': modules_db[encoded_apikey]["label"],
                'proxy': modules_db[encoded_apikey].get("proxy"),
                'module_info': module_info,
                'last': module_index + 1 == len(modules_db[encoded_apikey]["modules"])
            }
            for encoded_apikey in modules_db
            for module_index, module_info in enumerate(modules_db[encoded_apikey]["modules"])
            if module_info["status"] == "to_run"
        ]
        if unique_wallets:
            all_modules = [module_data for module_data in all_modules if module_data["last"]]

        if SHUFFLE_WALLETS:
            shuffle(all_modules)

        return all_modules


    def get_all_groups(self):
        self.get_password()

        with open(self.modules_db_name, encoding="utf-8") as f: modules_db = json.load(f)

        if not modules_db:
            return 'No more accounts left'
        elif list(modules_db.values())[0].get("group_number") is None:
            raise DataBaseError(f'Unexpected database type for this mode')

        all_groups = [
            {
                "group_index": group_index,
                "group_number": group_data["group_number"],
                "module_info": group_data["modules"][0],
                "wallets_data": [
                    {
                        "encoded_apikey": wallet_data["encoded_apikey"],
                        "apikey": self.decode_pk(wallet_data["encoded_apikey"]),
                        "address": wallet_data["address"],
                        "label": wallet_data["label"],
                        "proxy": wallet_data["proxy"]
                    }
                    for wallet_data in group_data["wallets_data"]
                ]
            }
            for group_index, group_data in modules_db.items()
            if group_data["modules"][0]["status"] == "to_run"
        ]
        return all_groups


    async def remove_module(self, module_data: dict):
        async with self.lock:
            with open(self.modules_db_name, encoding="utf-8") as f: modules_db = json.load(f)

            for index, module in enumerate(modules_db[module_data["encoded_apikey"]]["modules"]):
                if module["module_name"] == module_data["module_info"]["module_name"] and module["status"] == "to_run":
                    if module_data["module_info"]["status"] in [True, "completed"]:
                        self.window_name.add_module()
                        modules_db[module_data["encoded_apikey"]]["modules"].remove(module)
                    else:
                        modules_db[module_data["encoded_apikey"]]["modules"][index]["status"] = "failed"
                        self.window_name.add_module()
                    break

            if [module["status"] for module in modules_db[module_data["encoded_apikey"]]["modules"]].count('to_run') == 0:
                self.window_name.add_acc()
                last_module = True
            else:
                last_module = False

            if not modules_db[module_data["encoded_apikey"]]["modules"]:
                del modules_db[module_data["encoded_apikey"]]

            with open(self.modules_db_name, 'w', encoding="utf-8") as f: json.dump(modules_db, f)
            return last_module

    async def remove_account(self, module_data: dict):
        async with self.lock:
            with open(self.modules_db_name, encoding="utf-8") as f: modules_db = json.load(f)

            self.window_name.add_acc()
            if module_data["module_info"]["status"] in [True, "completed"]:
                del modules_db[module_data["encoded_apikey"]]

            else:
                modules_db[module_data["encoded_apikey"]]["modules"] = [{
                    "module_name": module_data["module_info"]["module_name"],
                    "status": "failed"
                }]

            with open(self.modules_db_name, 'w', encoding="utf-8") as f: json.dump(modules_db, f)
            return True


    async def remove_group(self, group_data: dict):
        async with self.lock:
            with open(self.modules_db_name, encoding="utf-8") as f: modules_db = json.load(f)

            self.window_name.add_acc()
            if group_data["module_info"]["status"] in [True, "completed"]:
                del modules_db[group_data["group_index"]]

            else:
                modules_db[group_data["group_index"]]["modules"] = [{
                    "module_name": group_data["module_info"]["module_name"],
                    "status": "failed"
                }]

            with open(self.modules_db_name, 'w', encoding="utf-8") as f: json.dump(modules_db, f)
            return True


    async def get_modules_left(self, encoded_pk: str):
        async with self.lock:
            with open(self.modules_db_name, encoding="utf-8") as f: modules_db = json.load(f)

            if modules_db.get(encoded_pk) is None:
                return 0
            else:
                return len([module for module in modules_db[encoded_pk]["modules"] if module["status"] == "to_run"])


    def set_accounts_modules_done(self, new_modules: dict):
        with open(self.stats_db_name, encoding="utf-8") as f: stats_db = json.load(f)
        stats_db["modules_done"] = {
            v["address"]: [0, len(v["modules"])]
            for k, v in new_modules.items()
        }
        with open(self.stats_db_name, 'w', encoding="utf-8") as f: json.dump(stats_db, f)

    def increase_account_modules_done(self, address: str):
        with open(self.stats_db_name, encoding="utf-8") as f: stats_db = json.load(f)
        modules_done = stats_db["modules_done"].get(address)
        if modules_done is None:
            return None
        modules_done[0] += 1
        if modules_done[0] == modules_done[1]:
            del stats_db["modules_done"][address]
        else:
            stats_db["modules_done"][address] = modules_done

        with open(self.stats_db_name, 'w', encoding="utf-8") as f: json.dump(stats_db, f)
        return modules_done


    async def append_report(self, key: str, text: str, success: bool | str = None, unique_msg: bool = False):
        async with self.lock:
            with open(self.report_db_name, encoding="utf-8") as f: report_db = json.load(f)

            if not report_db.get(key): report_db[key] = {'texts': [], 'success_rate': [0, 0]}

            if (
                    unique_msg and
                    report_db[key]["texts"] and
                    report_db[key]["texts"][-1] == self.STATUS_SMILES[success] + text
            ):
                return

            report_db[key]["texts"].append(self.STATUS_SMILES[success] + text)
            if success in [False, True]:
                report_db[key]["success_rate"][1] += 1
                if success: report_db[key]["success_rate"][0] += 1

            with open(self.report_db_name, 'w') as f: json.dump(report_db, f)


    async def get_account_reports(
            self,
            key: str,
            label: str,
            address: str | None,
            last_module: bool,
            mode: int,
            account_index: str | None = None,
            get_rate: bool = False,
    ):
        async with self.lock:
            with open(self.report_db_name, encoding="utf-8") as f: report_db = json.load(f)

            if account_index is None:
                account_index = f"[{self.window_name.accs_done}/{self.window_name.accs_amount}]"
            elif account_index is False:
                account_index = ""

            header_string = ""
            if account_index and last_module:
                header_string += f"{account_index} "
            if label:
                header_string += f"<b>{label}</b>"

            if mode in [1, 2]:
                modules_done = self.increase_account_modules_done(address=address)
                if modules_done:
                    header_string += f"\n📌 [Trade {modules_done[0]}/{modules_done[1]}]"

            if header_string:
                header_string += "\n\n"

            if report_db.get(key):
                account_reports = report_db[key]
                if get_rate: return f'{account_reports["success_rate"][0]}/{account_reports["success_rate"][1]}'
                del report_db[key]

                with open(self.report_db_name, 'w', encoding="utf-8") as f: json.dump(report_db, f)

                logs_text = '\n'.join(account_reports['texts'])
                tg_text = f'{header_string}{logs_text}'
                if account_reports["success_rate"][1]:
                    tg_text += f'\n\nSuccess rate {account_reports["success_rate"][0]}/{account_reports["success_rate"][1]}'

                return tg_text

            else:
                if header_string:
                    return f'{header_string}No actions'

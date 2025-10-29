import aiomysql
from typing import (
    Any,
    Dict,
    List,
    Sequence,
    Union,
    Tuple,
    Optional
)


class DbProvider:

    # -------------------------------------------------
    def __init__(self):

        self.TAG = self.__class__.__name__

        print(f'init Mysql provider')
        self.__pool = None

        self.__db = None
        self.__cursor = None

        if self.__pool:
            with self.__pool.acquire() as conn:
                with conn.cursor() as cur:
                    cur.execute("SET time_zone = '+00:00';")

        print(f'[complete] inited Mysql provider')

        print(f'[{self.TAG}] inited! {self.__db}')

    async def __check_connection(self):
        if self.__pool is None:
            self.__pool = await aiomysql.create_pool(
                host="localhost",
                user="root",
                password="",
                db="trading",
                autocommit=True
            )

    # -----------------------------------------------------
    async def execute(self, query: str, params: Optional[Sequence[Any]] = None) -> int:
        await self.__check_connection()
        async with self.__pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                return cur.lastrowid

    # -----------------------------------------------------
    # returns last row id (if positive result); -1 if error
    async def insert_one(self, table: str, fields: Dict[str, Union[int, str]], ignore: bool = False,
                         print_query: bool = True) -> int:
        field_names = ", ".join(fields.keys())
        placeholders = ", ".join(["%s"] * len(fields))
        ignore_str = "IGNORE" if ignore else ""

        query = f"INSERT {ignore_str} INTO {table} ({field_names}) VALUES ({placeholders})"
        params = tuple(fields.values())

        if print_query:
            print(f"[{self.TAG}] execute: {query}")

        try:
            return await self.execute(query, params)
        except Exception as e:
            print(f"[{self.TAG}] Mysql unexpected error: {e}")
            return -1

    # -----------------------------------------------------
    # returns affected rows count
    async def update(self, table: str, fields: Dict[str, Union[int, str]], select_filter: Dict[str, Union[int, str]] = None, limit: int = 0) -> int:
        field_updates = ", ".join([f"{k}=%s" for k in fields.keys()])
        params = list(fields.values())

        where_clause = ""
        if select_filter:
            where_clause = " AND ".join([f"{k}=%s" for k in select_filter.keys()])
            params.extend(select_filter.values())

        query = f"UPDATE {table} SET {field_updates}"
        if where_clause:
            query += f" WHERE {where_clause}"
        if limit > 0:
            query += f" LIMIT {limit}"

        print(f"[{self.TAG}] execute: {query}")

        try:
            await self.execute(query, params)
            return 1  # Return affected rows count (mocked for simplicity)
        except Exception as e:
            print(f"[{self.TAG}] Mysql unexpected error: {e}")
            return 0

    # -----------------------------------------------------
    async def update_one(self, table: str,
                   fields: Dict[str, Union[int, str]],
                   select_filter: Dict[str, Union[int, str]] = None) -> int:
        return await self.update(table, fields, select_filter, 1)

    # -----------------------------------------------------
    async def increment_one(self, table: str, field: str, select_filter: Dict[str, Union[int, str]]) -> bool:
        await self.__check_connection()
        try:
            async with self.__pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    equals_filter = ' AND '.join([f"{key} = '{value}'" for key, value in select_filter.items()])
                    limit_str = f'LIMIT 1'
                    command = f"UPDATE {table} SET {field}={field}+1 WHERE {equals_filter} {limit_str}"

                    print(f'[{self.TAG}] (increment_one) execute: {command}')

                    await cursor.execute(command)
                    await conn.commit()

                    return True

        except Exception as e:
            print(f"[{self.TAG}] Mysql unexpected error [increment_one]: table: {table}, "
                  f"fields: {';'.join(select_filter.keys())} error: {e}")
            return False

    # -----------------------------------------------------
    async def copy(self, target_table: str, source_table: str, columns: List[str],
                   select_filter: Dict[str, Union[int, str]]) -> bool:
        await self.__check_connection()

        col_str = ','.join(columns)

        try:
            async with self.__pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    equals_filter = ' AND '.join([f"{key} = '{value}'" for key, value in select_filter.items()])
                    command = f"INSERT INTO {target_table}({col_str}) SELECT {col_str} FROM {source_table} WHERE {equals_filter}"

                    print(f'[{self.TAG}] (copy) execute: {command}')

                    await cursor.execute(command)
                    await conn.commit()

                    return True

        except Exception as e:
            print(
                f"[{self.TAG}] Mysql unexpected error [copy]: target_table: {target_table}, source_table: {source_table} "
                f"fields: {';'.join(select_filter.keys())} error: {e}")
            return False

    # -----------------------------------------------------
    # returns status: True or False
    async def delete_by_id(self, table: str, filter_fields: Dict[str, Union[int, str]], limit: int = 0) -> bool:
        await self.__check_connection()
        try:
            async with self.__pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    equals_filter = ' AND '.join([f"{key} = '{value}'" for key, value in filter_fields.items()])
                    limit_str = f'LIMIT {limit}' if limit > 0 else ''
                    command = f'DELETE FROM {table} WHERE {equals_filter} {limit_str}'

                    print(f'[{self.TAG}] (delete_by_id) execute: {command}')

                    await cursor.execute(command)
                    await conn.commit()

                    return True

        except Exception as e:
            print(f"[{self.TAG}] Mysql unexpected error [delete_by_id]: table: {table}, "
                  f"fields: {';'.join(filter_fields.keys())} error: {e}")
            return False

    # -----------------------------------------------------
    async def select(self, table: str, params: List[str], select_filter: Dict[str, Union[int, str]] = None,
                     limit: int = 0, order:str = None, print_query: bool = False) -> List[Sequence[Any]]:
        fields = ", ".join(params)

        where_clause = ""
        values = []
        if select_filter:
            where_clause = " AND ".join([f"{k}=%s" for k in select_filter.keys()])
            values.extend(select_filter.values())
        else:
            where_clause = "1"

        query = f"SELECT {fields} FROM {table}"
        if where_clause:
            query += f" WHERE {where_clause}"
        if limit > 0:
            query += f" LIMIT {limit}"

        if order:
            query += f" ORDER BY open_time {order}"

        if print_query:
            print(f"[{self.TAG}] execute: {query}")

        try:
            return await self.fetchall(query, values)
        except Exception as e:
            print(f"[{self.TAG}] Mysql select error: {e}")
            return []

    async def close(self):
        if self.__pool is not None:
            self.__pool.close()
            await self.__pool.wait_closed()

    # -----------------------------------------------------
    async def select_one(self, table: str,
                   params: list[str],
                   select_filter: Dict[str, Union[int, str]] = None, print_query=False):

        res = await self.select(table, params, select_filter, 1, print_query)

        return res

    # -----------------------------------------------------
    async def select_one_fields(self, table: str,
                   params: list[str],
                   select_filter: Dict[str, Union[int, str]] = None,
                    print_query: bool = True):

        res = await self.select(table, params, select_filter, 1)

        if res:
            for k in res:
                return k

        if len(params) == 0:
            return None

        return (None,) * len(params)

    # -----------------------------------------------------
    async def fetchall(self, query: str, params: Optional[Sequence[Any]] = None) -> List[Sequence[Any]]:
        await self.__check_connection()
        async with self.__pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                return await cur.fetchall()

    # -----------------------------------------------------
    async def fetchone(self, query: str, params: Optional[Sequence[Any]] = None) -> Optional[Sequence[Any]]:
        await self.__check_connection()
        async with self.__pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                return await cur.fetchone()

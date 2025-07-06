"""Result型 - 関数型プログラミングのエラーハンドリング"""

from __future__ import annotations

import functools
from typing import Any, Awaitable, Callable, Generic, TypeVar, Union

T = TypeVar("T")
E = TypeVar("E")
U = TypeVar("U")


class Result(Generic[T, E]):
    """関数型プログラミングのResult型（Either型の実装）"""

    def __init__(self, value: Union[T, E], is_success: bool) -> None:
        self._value = value
        self._is_success = is_success

    @classmethod
    def success(cls, value: T) -> Result[T, E]:
        """成功を表すResultを作成"""
        return cls(value, True)

    @classmethod
    def failure(cls, error: E) -> Result[T, E]:
        """失敗を表すResultを作成"""
        return cls(error, False)

    def is_success(self) -> bool:
        """成功かどうかを判定"""
        return self._is_success

    def is_failure(self) -> bool:
        """失敗かどうかを判定"""
        return not self._is_success

    def unwrap(self) -> T:
        """値を取得（失敗の場合は例外）"""
        if self._is_success:
            return self._value  # type: ignore[return-value]
        msg = f"Result is failure: {self._value}"
        raise ValueError(msg)

    def unwrap_or(self, default: T) -> T:
        """値を取得（失敗の場合はデフォルト値）"""
        if self._is_success:
            return self._value  # type: ignore[return-value]
        return default

    def map(self, func: Callable[[T], U]) -> Result[U, E]:
        """成功の場合のみ関数を適用"""
        if self._is_success:
            try:
                return Result.success(func(self._value))  # type: ignore[arg-type]
            except Exception as e:
                return Result.failure(e)  # type: ignore[arg-type]
        return Result.failure(self._value)  # type: ignore[arg-type]

    def bind(self, func: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """成功の場合のみ関数を適用（モナドのbind）"""
        if self._is_success:
            try:
                return func(self._value)  # type: ignore[arg-type]
            except Exception as e:
                return Result.failure(e)  # type: ignore[arg-type]
        return Result.failure(self._value)  # type: ignore[arg-type]

    async def bind_async(self, func: Callable[[T], Awaitable[Result[U, E]]]) -> Result[U, E]:
        """非同期版のbind"""
        if self._is_success:
            try:
                return await func(self._value)  # type: ignore[arg-type]
            except Exception as e:
                return Result.failure(e)  # type: ignore[arg-type]
        return Result.failure(self._value)  # type: ignore[arg-type]

    async def map_async(self, func: Callable[[T], Awaitable[U]]) -> Result[U, E]:
        """非同期版のmap"""
        if self._is_success:
            try:
                result = await func(self._value)  # type: ignore[arg-type]
                return Result.success(result)
            except Exception as e:
                return Result.failure(e)  # type: ignore[arg-type]
        return Result.failure(self._value)  # type: ignore[arg-type]

    def map_error(self, func: Callable[[E], U]) -> Result[T, U]:
        """失敗の場合のみエラーを変換"""
        if self._is_failure:
            try:
                return Result.failure(func(self._value))  # type: ignore[arg-type]
            except Exception as e:
                return Result.failure(e)  # type: ignore[arg-type]
        return Result.success(self._value)  # type: ignore[arg-type]

    def match(
        self,
        success_func: Callable[[T], U],
        failure_func: Callable[[E], U],
    ) -> U:
        """パターンマッチング"""
        if self._is_success:
            return success_func(self._value)  # type: ignore[arg-type]
        return failure_func(self._value)  # type: ignore[arg-type]

    def pipe(self, func: Callable[[Result[T, E]], Result[U, E]]) -> Result[U, E]:
        """パイプライン処理 - Result型を受け取って別のResult型を返す関数を適用"""
        return func(self)

    def tap(self, func: Callable[[T], None]) -> Result[T, E]:
        """副作用のある処理（ログ出力など）を実行してそのまま返す"""
        if self._is_success:
            try:
                func(self._value)  # type: ignore[arg-type]
            except Exception:
                pass  # tapでの例外は無視
        return self

    def tap_error(self, func: Callable[[E], None]) -> Result[T, E]:
        """エラー時の副作用処理（エラーログ出力など）"""
        if self._is_failure:
            try:
                func(self._value)  # type: ignore[arg-type]
            except Exception:
                pass  # tapでの例外は無視
        return self

    # 演算子オーバーロード
    def __rshift__(self, func: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """>> 演算子でbind操作"""
        return self.bind(func)

    def __or__(self, default: T) -> T:
        """| 演算子でデフォルト値指定"""
        return self.unwrap_or(default)

    def __str__(self) -> str:
        if self._is_success:
            return f"Success({self._value})"
        return f"Failure({self._value})"

    def __repr__(self) -> str:
        return self.__str__()


def try_catch(func: Callable[[], T]) -> Result[T, Exception]:
    """例外をキャッチしてResult型に変換"""
    try:
        return Result.success(func())
    except Exception as e:
        return Result.failure(e)


def try_catch_async(
    func: Callable[[], Any]
) -> Callable[[], Result[Any, Exception]]:
    """非同期関数用の例外キャッチ"""
    async def wrapper() -> Result[Any, Exception]:
        try:
            result = await func()
            return Result.success(result)
        except Exception as e:
            return Result.failure(e)
    return wrapper


# デコレータパターン
def result_async(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[Result[T, Exception]]]:
    """非同期関数をResult型でラップするデコレータ"""
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Result[T, Exception]:
        try:
            result = await func(*args, **kwargs)
            return Result.success(result)
        except Exception as e:
            return Result.failure(e)
    return wrapper


def result_sync(func: Callable[..., T]) -> Callable[..., Result[T, Exception]]:
    """同期関数をResult型でラップするデコレータ"""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Result[T, Exception]:
        try:
            result = func(*args, **kwargs)
            return Result.success(result)
        except Exception as e:
            return Result.failure(e)
    return wrapper


# パイプライン処理のヘルパー関数
def pipe(*funcs: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """複数の関数をパイプライン処理するヘルパー関数"""
    def pipeline(value: Any) -> Any:
        for func in funcs:
            if isinstance(value, Result) and value.is_failure():
                return value
            value = func(value)
        return value
    return pipeline


def compose(*funcs: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """関数合成のヘルパー関数（右から左へ適用）"""
    def composition(value: Any) -> Any:
        for func in reversed(funcs):
            if isinstance(value, Result) and value.is_failure():
                return value
            value = func(value)
        return value
    return composition


# 便利なヘルパー関数
async def async_pipe(*funcs: Callable[[Any], Awaitable[Any]]) -> Callable[[Any], Awaitable[Any]]:
    """非同期パイプライン処理"""
    async def pipeline(value: Any) -> Any:
        for func in funcs:
            if isinstance(value, Result) and value.is_failure():
                return value
            value = await func(value)
        return value
    return pipeline


def when(condition: bool, then_func: Callable[[T], Result[U, E]], else_func: Callable[[T], Result[U, E]]) -> Callable[[T], Result[U, E]]:
    """条件分岐のヘルパー"""
    def conditional(value: T) -> Result[U, E]:
        if condition:
            return then_func(value)
        else:
            return else_func(value)
    return conditional


# 実用的な関数型プログラミングの例
class FunctionalPipeline:
    """関数型プログラミングのパイプライン処理の実用例"""
    
    @staticmethod
    def validate_and_transform(data: str) -> Result[dict, str]:
        """データ検証と変換のパイプライン例"""
        return (
            Result.success(data)
            .map(lambda x: x.strip())
            .bind(lambda x: Result.failure("Empty string") if not x else Result.success(x))
            .map(lambda x: {"value": x, "length": len(x)})
            .tap(lambda x: print(f"Processed: {x}"))
        )
    
    @staticmethod
    async def async_data_pipeline(query: str) -> Result[dict, Exception]:
        """非同期データ処理パイプラインの例"""
        import asyncio
        
        async def fetch_data(q: str) -> dict:
            await asyncio.sleep(0.1)  # 疑似的な非同期処理
            return {"query": q, "result": f"processed_{q}"}
        
        async def validate_data(data: dict) -> dict:
            if not data.get("query"):
                raise ValueError("Invalid query")
            return data
        
        return await (
            Result.success(query)
            .map_async(fetch_data)
            .bind_async(lambda data: Result.success(data).map_async(validate_data))
            .tap(lambda result: print(f"Pipeline result: {result}"))
        )


# 使用例をコメントで示す
"""
使用例:

# 1. 基本的なメソッドチェーン
result = (
    Result.success("  hello world  ")
    .map(str.strip)
    .map(str.upper)
    .tap(lambda x: print(f"Processed: {x}"))
    .unwrap_or("DEFAULT")
)

# 2. 演算子オーバーロード
result = (
    Result.success(10)
    >> (lambda x: Result.success(x * 2))
    >> (lambda x: Result.success(x + 1))
) | 0  # デフォルト値

# 3. パターンマッチング
message = result.match(
    success_func=lambda x: f"Success: {x}",
    failure_func=lambda e: f"Error: {e}"
)

# 4. エラーハンドリング付きパイプライン
result = (
    user_input
    .bind(validate_input)
    .bind(process_data)
    .bind(save_result)
    .tap_error(lambda e: logger.error(f"Pipeline failed: {e}"))
    .map_error(lambda e: UserFriendlyError(str(e)))
)

# 5. 非同期処理
async def async_example():
    result = await (
        Result.success("query")
        .map_async(fetch_from_api)
        .bind_async(validate_response)
        .map_async(transform_data)
    )
    return result | {}  # デフォルト値
""" 
"""Result型 - 関数型プログラミングのエラーハンドリング"""

from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar, Union

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


# パイプライン処理のヘルパー関数
def pipe(*funcs: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """複数の関数をパイプライン処理するヘルパー関数
    
    使用例:
    result = (
        initial_value
        |> pipe(
            transform_data,
            validate_data,
            save_data
        )
    )
    """
    def pipeline(value: Any) -> Any:
        for func in funcs:
            if isinstance(value, Result) and value.is_failure():
                return value
            value = func(value)
        return value
    return pipeline


def compose(*funcs: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """関数合成のヘルパー関数（右から左へ適用）
    
    使用例:
    process = compose(save_data, validate_data, transform_data)
    result = process(initial_value)
    """
    def composition(value: Any) -> Any:
        for func in reversed(funcs):
            if isinstance(value, Result) and value.is_failure():
                return value
            value = func(value)
        return value
    return composition 
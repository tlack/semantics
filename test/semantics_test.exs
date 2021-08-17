defmodule SemanticsTest do
  use ExUnit.Case
  doctest Semantics

  test "greets the world" do
    assert Semantics.hello() == :world
  end
end

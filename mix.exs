defmodule Semantics.MixProject do
  use Mix.Project

  def project do
    [
      app: :semantics,
      version: "0.1.0",
      elixir: "~> 1.12",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:similarity, "~> 0.2"},
      {:erlport, "~> 0.9"},
    ]
  end
end

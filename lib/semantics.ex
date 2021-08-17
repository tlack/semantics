defmodule Semantics do
  @moduledoc """
  Semantic similarity tools using sentence embeddings.
  
  Semantics is an Elixir wrapper around the Python library SentenceTransformers
  by [SBert.net](http://sbert.net)

  """

  use GenServer
 
  @default_model "paraphrase-MiniLM-L6-v2"

  # XXX :code.priv_dir() not available during compilation?
  IO.puts(__DIR__)
  @python_dir Path.expand("#{__DIR__}/../priv/python")
  @venv_path Path.join([@python_dir, "semantics-venv"])
  @python_bin Path.join([@venv_path, "bin", "python3"])
  # IO.inspect(@python_dir, label: "Semantics @python_dir")
  # IO.inspect(@venv_path, label: "Semantics @venv_path")
  if not File.exists?(@venv_path) do
    IO.puts("🚨 SEMANTICS - First start - Configuring VENV in #{@venv_path}")
    cmd = ["python3", "-m", "venv", @venv_path]
    IO.puts("🚨 SEMANTICS - Configuring venv")
    IO.puts(Enum.join(cmd, " "))
    
    {out_text, err_code} = System.cmd(hd(cmd), tl(cmd))

    if err_code != 0 do
      IO.puts("🚨 SEMANTICS - Error creating venv. Output:")
      IO.puts(out_text)
      raise "semantics could not create venv"
    else
      IO.puts("🚨 SEMANTICS - Installing requirements. This may take some time.")

      reqs_path = Path.join([@python_dir, "requirements.txt"])
      cmd = [@python_bin, "-m", "pip", "install", "-r", reqs_path]
      IO.puts(Enum.join(cmd, " "))

      {out_text, err_code} = System.cmd(hd(cmd), tl(cmd), env: [{"VIRTUAL_ENV", @venv_path}])
      if err_code != 0 do
        IO.puts("🚨 SEMANTICS - Error installing requirements. Output:")
        IO.puts(out_text)
        raise "semantics could not create venv"
      end
    end
  end

  # Interface

  def start_link() do
    start_link(@default_model)
  end
  def start_link(model) do
    GenServer.start_link(__MODULE__, model, name: Semantics)
  end
  
  @doc """
  Retrieve embedding vector for a given text string with the currently loaded model

  ## Examples

      iex> Semantics.start_link("paraphrase-MiniLM-L6-v2")
      {:ok, PID<0.0>}
      iex> Semantics.embedding("I love geckos")
      [-0.6230819821357727, -1.1321643590927124, 0.20272356271743774,
      0.08726023882627487, -0.6386743187904358, -0.04986400529742241,
      0.7474567890167236, 0.04601148143410683, 0.07410695403814316,
      0.36048224568367004, -0.7157518267631531, -0.9080777168273926,
      -0.3780222237110138, 0.22153961658477783, -0.2679588198661804,
      0.10182621330022812, 0.45531317591667175, -0.3617912828922272,
      0.0209545586258173, 0.0662737488746643, -0.5444900393486023,
      -0.30246955156326294, -0.49750199913978577, 0.594270646572113,
      -0.4451166093349457, -0.1890772581100464, -0.8081623911857605,
      0.4554007053375244, -0.1811652034521103, -0.12635637819766998,
      0.35992708802223206, -0.4459587335586548, 0.6338019371032715,
      -0.23431335389614105, -0.5372543931007385, 0.32740703225135803,
      -0.03602148965001106, -0.4820868670940399, -0.2127869874238968,
      -0.12680721282958984, 0.023507649078965187, 0.21502859890460968,
      0.22868287563323975, 0.15201695263385773, -0.009994926862418652,
      -0.6130571961402893, -0.18206633627414703, -0.7062084674835205,
      0.66861891746521, 0.7331072688102722, ...]
  """
  def embedding(text) do
    {:ok, result} = GenServer.call(Semantics, {:predict, text})
    result
  end

  @doc """
  Retrieve embedding vector for a given text string from a given named model

  ## Examples

      iex> Semantics.embedding("I love geckos", "paraphrase-MiniLM-L6-v2")
      [-0.6230819821357727, -1.1321643590927124, 0.20272356271743774,
      0.08726023882627487, -0.6386743187904358, -0.04986400529742241,
      0.7474567890167236, 0.04601148143410683, 0.07410695403814316,
      0.36048224568367004, -0.7157518267631531, -0.9080777168273926,
      -0.3780222237110138, 0.22153961658477783, -0.2679588198661804,
      0.10182621330022812, 0.45531317591667175, -0.3617912828922272,
      0.0209545586258173, 0.0662737488746643, -0.5444900393486023,
      -0.30246955156326294, -0.49750199913978577, 0.594270646572113,
      -0.4451166093349457, -0.1890772581100464, -0.8081623911857605,
      0.4554007053375244, -0.1811652034521103, -0.12635637819766998,
      0.35992708802223206, -0.4459587335586548, 0.6338019371032715,
      -0.23431335389614105, -0.5372543931007385, 0.32740703225135803,
      -0.03602148965001106, -0.4820868670940399, -0.2127869874238968,
      -0.12680721282958984, 0.023507649078965187, 0.21502859890460968,
      0.22868287563323975, 0.15201695263385773, -0.009994926862418652,
      -0.6130571961402893, -0.18206633627414703, -0.7062084674835205,
      0.66861891746521, 0.7331072688102722, ...]

  """
  def embedding(text, model) do
    {:ok, result} = GenServer.call(Semantics, {:predict, text, model})
    result
  end

  def load(model) do
    GenServer.call(Semantics, {:load, model})
  end

  def similarity(a, b, [type: "cosine"]) do
    Similarity.cosine(a, b) 
  end

  def similarity(a, b) do
    similarity(a, b, [type: "cosine"])
  end

  # Implementation

  @impl true
  def init(model) do
    args = [
      {:env, [{'VIRTUAL_ENV', to_charlist(@venv_path)}]},
      {:python, to_charlist(@python_bin)}, 
      {:python_path, to_charlist(@python_dir)}
    ]
      |> IO.inspect(label: "python start args")
    {:ok, pid} = :python.start(args)

    :python.call(pid, :app, :load, [model])

    {:ok, {model, pid}}
  end

  @impl true
  def handle_call({:load, model}, _from, {_, pid}) do
    :python.call(pid, :app, :load, [model])
    {:reply, {:ok, model}, {model, pid}}
  end

  @impl true
  def handle_call({:predict, text}, from, {model, pid}) do
    handle_call({:predict, text, model}, from, {model, pid})
  end

  @impl true
  def handle_call({:predict, text, model}, _from, {_, pid}) do
    # IO.inspect({:predict, text, from, pid}, label: :predict)
    resp = :python.call(pid, :app, :predict, [model, text])
    {:reply, {:ok, resp}, {model, pid}}
  end

end

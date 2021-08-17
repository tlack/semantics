defmodule SemanticsTest do
  use ExUnit.Case
  
  @test_model "paraphrase-MiniLM-L6-v2"
  @test_model_b "paraphrase-MiniLM-L3-v2"

  test "single embeddings" do
    assert {:ok, pid} = Semantics.start_link(@test_model)
    assert embedding = Semantics.embedding("I love kittens")
    IO.inspect(embedding, label: "embedding")
    assert length(embedding) == 384
    assert Enum.at(embedding, 0) != 0.0
  end

  test "comparison w/ similarity()" do
    assert {:ok, pid} = Semantics.start_link(@test_model)
    assert emb1 = Semantics.embedding("I love kittens")
    assert emb2 = Semantics.embedding("I love cats")
    assert emb3 = Semantics.embedding("I love helium balloons")
    assert Semantics.similarity(emb1, emb2) > Semantics.similarity(emb2, emb3) 
  end
 
end

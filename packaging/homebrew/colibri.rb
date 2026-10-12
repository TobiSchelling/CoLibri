class Colibri < Formula
  desc "Local RAG system for semantic search over markdown content"
  homepage "https://github.com/TobiSchelling/CoLibri"
  version "0.16.2"
  license "MIT"

  on_macos do
    on_arm do
      url "https://github.com/TobiSchelling/CoLibri/releases/download/v#{version}/colibri-#{version}-macos-arm64.tar.gz"
      sha256 "76fe910cd51ce85cec9f1fce7648814e2dd3f1a9ad76b96bf0180f5597815b36"
    end
  end

  def install
    bin.install "colibri"
  end

  def caveats
    <<~EOS
      CoLibri requires Ollama for local embeddings.

      Install and start Ollama:
        brew install ollama
        ollama serve
        ollama pull bge-m3

      Then verify installation:
        colibri doctor

      Configure sources in ~/.config/colibri/config.yaml
    EOS
  end

  test do
    assert_match "colibri", shell_output("#{bin}/colibri --help")
  end
end

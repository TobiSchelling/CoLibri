class Colibri < Formula
  desc "Local RAG system for semantic search over markdown content"
  homepage "https://github.com/TobiSchelling/CoLibri"
  version "0.3.0"
  license "MIT"

  on_macos do
    on_arm do
      url "https://github.com/TobiSchelling/CoLibri/releases/download/v#{version}/colibri-#{version}-macos-arm64.tar.gz"
      sha256 "02865a97c1f9fa36d797a42c721c048848e4610d8e9dc8cfb8a34c13ad69a0fd"
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
        ollama pull nomic-embed-text

      Then verify installation:
        colibri doctor

      Configure sources in ~/.config/colibri/config.yaml
    EOS
  end

  test do
    assert_match "colibri", shell_output("#{bin}/colibri --help")
  end
end

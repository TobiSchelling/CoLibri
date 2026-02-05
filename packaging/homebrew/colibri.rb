class Colibri < Formula
  include Language::Python::Virtualenv

  desc "Local-first RAG for technical books and notes (CoLibri)"
  homepage "https://gitlab.com/Tobias.Schelling/CoLibri"

  version "0.2.0"
  url "https://gitlab.com/Tobias.Schelling/CoLibri/-/releases/v#{version}/downloads/colibri-#{version}.tar.gz"
  sha256 "beae316a7e04307a683cb9da4222c235703a042aad18805d7b215653c96cde2d"

  license "MIT"

  depends_on "python@3.12"

  # If you add non-Python dependencies (e.g. poppler), declare them here.
  # depends_on "poppler"

  def install
    virtualenv_install_with_resources
  end

  def caveats
    <<~EOS
      CoLibri uses Ollama for local embeddings.

        brew install ollama
        ollama pull nomic-embed-text

      Then run:

        colibri doctor
        colibri index
    EOS
  end

  test do
    assert_match "CoLibri", shell_output("#{bin}/colibri --help")
  end
end

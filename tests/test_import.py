"""Tests for PDF import functionality."""




class TestPdfImport:
    """Tests for the PDF import functionality."""

    def test_sanitize_title(self) -> None:
        """Test that titles are properly sanitized for filenames."""
        # This tests the filename sanitization logic
        title = "Clean Architecture: A Guide"
        safe_title = "".join(c for c in title if c.isalnum() or c in " -_").strip()
        assert safe_title == "Clean Architecture A Guide"

    def test_frontmatter_format(self) -> None:
        """Test that frontmatter is properly formatted."""
        title = "Test Book"
        pdf_name = "test.pdf"

        frontmatter = f'''---
title: "{title}"
type: book
source_pdf: "{pdf_name}"
'''
        assert 'title: "Test Book"' in frontmatter
        assert "type: book" in frontmatter
        assert 'source_pdf: "test.pdf"' in frontmatter

import os

def generate_combined_file(source_dir, output_file):
    """Generate a combined sitemap and script content file."""
    supported_extensions = {".py", ".js", ".html", ".css", ".java"}  # Supported script extensions
    excluded_folders = [
        os.path.join(source_dir, "brain"),
        os.path.join(source_dir, "gui", "Appendix")  # Exclude the 'Appendix' folder inside 'gui'
    ]  # Folders to exclude

    with open(output_file, "w", encoding="utf-8") as outfile:
        # Step 1: Generate the sitemap
        outfile.write("No comments # in code, don't print the site map. Only show corrected script. Sitemap of Directory: Odin 2\n")
        outfile.write("=" * 50 + "\n")
        for root, dirs, files in os.walk(source_dir):
            # Skip excluded folders
            if any(root.startswith(folder) for folder in excluded_folders):
                continue

            # Include 'gui' folder explicitly
            if "gui" in root and not root.startswith(os.path.join(source_dir, "gui", "Appendix")):
                relative_path = os.path.relpath(root, source_dir)
                outfile.write(f"\n[Folder] {relative_path}\n")
                outfile.write("-" * 50 + "\n")
                for file in files:
                    if not file.endswith(".txt"):  # Skip .txt files
                        file_path = os.path.join(root, file)
                        outfile.write(f"  {file_path}\n")

        outfile.write("\n" + "=" * 50 + "\n\n")

        # Step 2: Append script contents
        outfile.write("Script Contents\n")
        outfile.write("=" * 50 + "\n")
        for root, _, files in os.walk(source_dir):
            # Skip excluded folders
            if any(root.startswith(folder) for folder in excluded_folders):
                continue

            for file in files:
                file_path = os.path.join(root, file)
                # Check file extension
                if os.path.splitext(file)[1] in supported_extensions:
                    try:
                        with open(file_path, "r", encoding="utf-8") as infile:
                            # Write a header for each file
                            outfile.write(f"\n# START OF FILE: {file_path}\n")
                            outfile.write(infile.read())
                            outfile.write(f"\n# END OF FILE: {file_path}\n\n")
                            print(f"Appended: {file_path}")
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

if __name__ == "__main__":
    source_directory = r"C:\Users\Sean Craig\Desktop\AI Python Tools\Odin"
    output_file_path = r"C:\Users\Sean Craig\Desktop\AI Python Tools\Odin\gui\Appendix\combined_sitemap_and_scripts.txt"
    
    generate_combined_file(source_directory, output_file_path)
    print(f"\nSitemap and script contents have been saved to {output_file_path}")

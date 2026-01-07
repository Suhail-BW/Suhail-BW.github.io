# Jekyll Local Installation Guide

Follow these steps to install Jekyll and test your site locally.

## Step 1: Install Ruby and Dependencies

```bash
sudo apt update
sudo apt install -y ruby-full build-essential zlib1g-dev
```

## Step 2: Configure Gem Installation Path

Add these lines to your `~/.bashrc` or `~/.zshrc`:

```bash
echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc
echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc
echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

## Step 3: Install Bundler

```bash
gem install bundler
```

## Step 4: Install Jekyll Dependencies

Navigate to your project directory:

```bash
cd /home/suhail/devel/Suhail-BW.github.io
bundle install
```

## Step 5: Run Jekyll Server

```bash
bundle exec jekyll serve
```

Or to make it accessible from other devices on your network:

```bash
bundle exec jekyll serve --host 0.0.0.0
```

## Step 6: View Your Site

Open your browser and go to:
- Local: http://localhost:4000
- Network: http://127.0.0.1:4000

Your site will auto-reload when you make changes to files!

## Common Issues

### Port Already in Use
If port 4000 is already in use:
```bash
bundle exec jekyll serve --port 4001
```

### Permission Errors
If you get permission errors, make sure you've set up GEM_HOME in Step 2.

### Build Errors
If you get build errors, try:
```bash
bundle update
bundle install
```

## Quick Commands Reference

- **Start server**: `bundle exec jekyll serve`
- **Build site**: `bundle exec jekyll build`
- **Clean build files**: `bundle exec jekyll clean`
- **Stop server**: Press `Ctrl+C` in the terminal

## Adding New Blog Posts

1. Create a new file in `_posts/` with format: `YYYY-MM-DD-title.md`
2. Add frontmatter at the top:
   ```yaml
   ---
   layout: post
   title: "Your Title"
   date: 2024-01-15
   description: "Brief description"
   ---
   ```
3. Write your content in Markdown
4. Save the file
5. Jekyll will automatically rebuild and show the new post!

The server watches for file changes and rebuilds automatically, so you can edit and see results immediately in your browser.

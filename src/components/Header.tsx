
import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Menu, X } from 'lucide-react';

const Header = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  return (
    <header className="sticky top-0 z-50 py-4 px-6 border-b border-white/20 dark:border-white/10 flex items-center justify-between backdrop-blur-md bg-white/70 dark:bg-black/70 shadow-sm transition-all duration-300">
      <Link to="/" className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-brand-purple to-brand-pink hover:opacity-80 transition-opacity">
        Jatin Wig
      </Link>

      {/* Desktop Navigation */}
      <div className="flex items-center gap-4">
        <nav className="hidden md:flex space-x-8">
          {[
            { name: "Home", path: "/" },
            { name: "Skills", path: "/skills" },
            { name: "Projects", path: "/projects" },
            { name: "Research", path: "/publications" },
            { name: "Experience", path: "/experience" },
            { name: "Resume", path: "/resume" },
            { name: "Contact", path: "/contact" },
          ].map((link, index) => (
            <Link
              key={link.path}
              to={link.path}
              className="relative text-foreground hover:text-brand-purple dark:hover:text-purple-300 transition-colors font-medium text-sm group"
              style={{ animation: `fade-in 0.5s ease-out ${index * 0.1}s backwards` }}
            >
              {link.name}
              <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-brand-purple transition-all group-hover:w-full" />
            </Link>
          ))}
        </nav>

        {/* Mobile Hamburger Button */}
        <div className="flex items-center gap-2 md:hidden">
          <button
            onClick={toggleMenu}
            className="p-2 rounded-md hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-200"
          >
            {isMenuOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>
      </div>

      {/* Mobile Navigation Menu */}
      {isMenuOpen && (
        <div className="absolute top-full left-0 right-0 bg-white/95 dark:bg-black/95 backdrop-blur-xl border-b border-gray-100 dark:border-gray-800 shadow-lg md:hidden z-50 animate-accordion-down">
          <nav className="flex flex-col py-4">
            {[
              { name: "Home", path: "/" },
              { name: "Skills", path: "/skills" },
              { name: "Projects", path: "/projects" },

              { name: "Research & Publications", path: "/publications" },
              { name: "Experience", path: "/experience" },
              { name: "Volunteering", path: "/volunteering" },
              { name: "Education", path: "/education" },
              { name: "Resume", path: "/resume" },
              { name: "Contact", path: "/contact" },
            ].map((link) => (
              <Link
                key={link.path}
                to={link.path}
                className="px-6 py-3 text-foreground hover:text-brand-purple dark:hover:text-purple-300 hover:bg-brand-purple/5 transition-colors font-medium border-l-4 border-transparent hover:border-brand-purple dark:hover:border-purple-300"
                onClick={toggleMenu}
              >
                {link.name}
              </Link>
            ))}
          </nav>
        </div>
      )}
    </header>
  );
};

export default Header;
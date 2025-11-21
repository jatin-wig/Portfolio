
import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Menu, X } from 'lucide-react';

const Header = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  return (
    <header className="py-4 px-6 borderd-b flex items-center justify-between">
      <Link to="/" className="text-2xl font-bold text-brand-purple">
        Jatin Wig
      </Link>
      
      {/* Desktop Navigation */}
      <div className="flex items-center">
        <nav className="hidden md:flex space-x-8">
          <Link to="/" className="text-foreground hover:text-brand-purple">Home</Link>
          <Link to="/skills" className="text-foreground hover:text-brand-purple">Skills</Link>
          <Link to="/projects" className="text-foreground hover:text-brand-purple">Projects</Link>
          <Link to="/genai-projects" className="text-foreground hover:text-brand-purple">GenAI Projects</Link>
          <Link to="/publications" className="text-foreground hover:text-brand-purple">Research & Publications</Link>
          <Link to="/experience" className="text-foreground hover:text-brand-purple">Experience</Link>
          <Link to="/volunteering" className="text-foreground hover:text-brand-purple">Volunteering</Link>
          <Link to="/education" className="text-foreground hover:text-brand-purple">Education</Link>
          <Link to="/resume" className="text-foreground hover:text-brand-purple">Resume</Link>
          <Link to="/contact" className="text-foreground hover:text-brand-purple">Contact</Link>
        </nav>
        
        {/* Mobile Hamburger Button */}
        <button 
          onClick={toggleMenu}
          className="md:hidden p-2 rounded-md hover:bg-gray-100"
        >
          {isMenuOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>

      {/* Mobile Navigation Menu */}
      {isMenuOpen && (
        <div className="absolute top-full left-0 right-0 bg-white border-b shadow-lg md:hidden z-50">
          <nav className="flex flex-col py-4">
            <Link to="/" className="px-6 py-3 text-foreground hover:text-brand-purple hover:bg-gray-50" onClick={toggleMenu}>Home</Link>
            <Link to="/skills" className="px-6 py-3 text-foreground hover:text-brand-purple hover:bg-gray-50" onClick={toggleMenu}>Skills</Link>
            <Link to="/projects" className="px-6 py-3 text-foreground hover:text-brand-purple hover:bg-gray-50" onClick={toggleMenu}>Projects</Link>
            <Link to="/genai-projects" className="px-6 py-3 text-foreground hover:text-brand-purple hover:bg-gray-50" onClick={toggleMenu}>GenAI Projects</Link>
            <Link to="/publications" className="px-6 py-3 text-foreground hover:text-brand-purple hover:bg-gray-50" onClick={toggleMenu}>Research & Publications</Link>
            <Link to="/experience" className="px-6 py-3 text-foreground hover:text-brand-purple hover:bg-gray-50" onClick={toggleMenu}>Experience</Link>
            <Link to="/volunteering" className="px-6 py-3 text-foreground hover:text-brand-purple hover:bg-gray-50" onClick={toggleMenu}>Volunteering</Link>
            <Link to="/education" className="px-6 py-3 text-foreground hover:text-brand-purple hover:bg-gray-50" onClick={toggleMenu}>Education</Link>
            <Link to="/resume" className="px-6 py-3 text-foreground hover:text-brand-purple hover:bg-gray-50" onClick={toggleMenu}>Resume</Link>
            <Link to="/contact" className="px-6 py-3 text-foreground hover:text-brand-purple hover:bg-gray-50" onClick={toggleMenu}>Contact</Link>
          </nav>
        </div>
      )}
    </header>
  );
};

export default Header;
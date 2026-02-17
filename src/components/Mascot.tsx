import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useLocation } from 'react-router-dom';

type MascotMessage = {
  text: string;
  emoji: string;
  duration?: number;
  color: string;
};

const Mascot: React.FC = () => {
  const [visible, setVisible] = useState(false);
  const [message, setMessage] = useState<MascotMessage | null>(null);
  const [hasShownWelcome, setHasShownWelcome] = useState(false);
  const location = useLocation();

  // Jatin's custom messages
  const messages: MascotMessage[] = [
    { text: "Still scrolling? Jatin saved the best for last 👇", emoji: "👇", color: "bg-gradient-to-r from-green-500 to-emerald-500", duration: 5000 },
    { text: "👨‍💻 Projects built with Jatin's late-night hustle and caffeine", emoji: "☕", color: "bg-gradient-to-r from-blue-500 to-cyan-500", duration: 6000 },
    { text: "Not hiring Jatin yet? That's a bold move... 🚩", emoji: "🚩", color: "bg-gradient-to-r from-red-500 to-pink-500", duration: 5000 },
    { text: "Wanna talk code, design, or dreams? Ping Jatin! 📨", emoji: "📨", color: "bg-gradient-to-r from-purple-500 to-violet-500", duration: 5000 },
    { text: "💼 Jatin's skills: sharper than your deadline stress", emoji: "💼", color: "bg-gradient-to-r from-indigo-500 to-blue-500", duration: 6000 },
    { text: "🔥 This portfolio? Built with more love than Jatin gave his ex 😅", emoji: "💔", color: "bg-gradient-to-r from-orange-500 to-red-500", duration: 6000 },
    { text: "If Jatin hasn't impressed you yet... hold his coffee ☕", emoji: "☕", color: "bg-gradient-to-r from-amber-500 to-orange-500", duration: 5000 },
    { text: "VS Code, ambition, and Jatin — the holy trinity of dev power ⚡", emoji: "⚡", color: "bg-gradient-to-r from-yellow-500 to-amber-500", duration: 6000 },
    { text: "👀 Pretend you're discovering the next tech prodigy — 'cause you are (hi 👋, Jatin here)", emoji: "👋", color: "bg-gradient-to-r from-emerald-500 to-teal-500", duration: 7000 },
    { text: "If Elon needed a dev, Jatin's phone would already be buzzing 🚀", emoji: "🚀", color: "bg-gradient-to-r from-violet-500 to-purple-500", duration: 6000 },
    { text: "Hiring Jatin = your smartest decision since version control 💾", emoji: "💾", color: "bg-gradient-to-r from-cyan-500 to-blue-500", duration: 6000 },
    { text: "📂 His resume slaps — go peek at it before someone else does!", emoji: "📂", color: "bg-gradient-to-r from-pink-500 to-rose-500", duration: 5000 },
    { text: "👾 Jatin's code is so clean, even linters applaud 👏", emoji: "👏", color: "bg-gradient-to-r from-teal-500 to-cyan-500", duration: 5000 },
    { text: "You found Jatin. Now find the 'Contact' button. Win-win.", emoji: "🎯", color: "bg-gradient-to-r from-lime-500 to-green-500", duration: 5000 },
  ];

  // Welcome message for website visits
  const welcomeMessage: MascotMessage = {
    text: "Hey, you! Welcome to Jatin's world of logic and pixels 🌐",
    emoji: "🌐",
    color: "bg-gradient-to-r from-indigo-500 to-purple-500",
    duration: 6000
  };

  // Show mascot with specific message
  const showMascot = (customMessage?: MascotMessage) => {
    if (visible) return; // Prevent multiple simultaneous messages

    const messageToShow = customMessage || messages[Math.floor(Math.random() * messages.length)];

    setMessage(messageToShow);
    setVisible(true);

    // Auto-hide after duration
    setTimeout(() => {
      setVisible(false);
    }, messageToShow.duration || 5000);
  };

  // Check if welcome message was already shown in this session
  useEffect(() => {
    const welcomeShown = localStorage.getItem('jatin-welcome-shown');
    if (!welcomeShown) {
      // Show welcome message after 2 seconds on first visit
      setTimeout(() => {
        showMascot(welcomeMessage);
        setHasShownWelcome(true);
        localStorage.setItem('jatin-welcome-shown', 'true');
      }, 2000);
    } else {
      setHasShownWelcome(true);
    }
  }, []);

  // Set up 3-minute intervals for random messages (only after welcome or if welcome was already shown)
  useEffect(() => {
    if (!hasShownWelcome) return;

    const intervalTimer = setInterval(() => {
      if (!visible) {
        showMascot();
      }
    }, 180000); // 3 minutes = 180,000ms

    return () => {
      clearInterval(intervalTimer);
    };
  }, [hasShownWelcome, visible]);

  // Remove the click handler that was showing welcome message on every click
  // This was causing too many pop-ups

  return (
    <AnimatePresence>
      {visible && message && (
        <motion.div
          initial={{ scale: 0, opacity: 0, x: 100, y: 100 }}
          animate={{
            scale: 1,
            opacity: 1,
            x: 0,
            y: 0,
          }}
          exit={{
            scale: 0,
            opacity: 0,
            x: 100,
            y: 100,
            transition: { duration: 0.8 }
          }}
          transition={{
            type: "spring",
            damping: 15,
            stiffness: 300,
          }}
          className="fixed bottom-4 right-4 md:bottom-8 md:right-8 z-50 pointer-events-auto"
        >
          {/* Main Character Container */}
          <div className="relative flex flex-col items-center">

            {/* Close Button */}
            <button
              onClick={(e) => { e.stopPropagation(); setVisible(false); }}
              className="absolute -top-2 -right-2 bg-white text-gray-500 rounded-full p-1 shadow-md hover:bg-red-50 hover:text-red-500 transition-colors z-[60]"
              aria-label="Close Mascot"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
            </button>

            {/* Speech bubble */}
            <motion.div
              initial={{ scale: 0, y: 20 }}
              animate={{ scale: 1, y: 0 }}
              transition={{ delay: 0.3, type: "spring", damping: 15 }}
              className="bg-white rounded-3xl px-6 py-4 shadow-2xl border-4 border-gray-200 min-w-[280px] max-w-[350px] mb-4 relative"
            >
              {/* Speech bubble arrow pointing down */}
              <div className="absolute -bottom-3 left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-[15px] border-r-[15px] border-t-[15px] border-l-transparent border-r-transparent border-t-white"></div>
              <div className="absolute -bottom-4 left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-[15px] border-r-[15px] border-t-[15px] border-l-transparent border-r-transparent border-t-gray-200"></div>

              <div className="flex items-center gap-4 text-gray-800">
                <motion.span
                  className="text-2xl"
                  animate={{ scale: [1, 1.2, 1], rotate: [0, 10, -10, 0] }}
                  transition={{ duration: 2, repeat: Infinity }}
                >
                  {message.emoji}
                </motion.span>
                <p className="text-sm font-bold leading-relaxed">{message.text}</p>
              </div>
            </motion.div>

            {/* Enhanced Animated Character Body with Custom Avatar */}
            <motion.div
              className={`relative ${message.color} rounded-full p-4 shadow-2xl border-4 border-white cursor-pointer hover:scale-105 transition-transform`}
              onClick={() => showMascot()}
              whileHover={{ rotate: [0, -5, 5, 0] }}
              animate={{
                y: [0, -15, 0],
                rotate: [0, 5, -5, 0]
              }}
              transition={{
                duration: 3,
                repeat: Infinity,
                ease: "easeInOut"
              }}
            >
              {/* Glowing aura effect */}
              <div className="absolute inset-0 rounded-full opacity-40 animate-pulse bg-white/50 blur-sm"></div>

              {/* Custom Avatar Face Based on Your Features */}
              <motion.div
                animate={{
                  scale: [1, 1.05, 1],
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  ease: "easeInOut"
                }}
                className="relative z-10 w-20 h-20 rounded-full overflow-hidden border-4 border-white shadow-lg bg-gradient-to-br from-amber-100 to-orange-200"
              >
                {/* Face Base */}
                <div className="w-full h-full relative">
                  {/* Hair */}
                  <div className="absolute top-0 left-0 w-full h-12 bg-gradient-to-b from-gray-800 to-gray-700 rounded-t-full"></div>

                  {/* Forehead */}
                  <div className="absolute top-8 left-0 w-full h-6 bg-gradient-to-b from-amber-200 to-orange-200"></div>

                  {/* Eyes */}
                  <motion.div
                    className="absolute top-10 left-3 w-2 h-2 bg-gray-800 rounded-full"
                    animate={{ scaleY: [1, 0.1, 1] }}
                    transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
                  />
                  <motion.div
                    className="absolute top-10 right-3 w-2 h-2 bg-gray-800 rounded-full"
                    animate={{ scaleY: [1, 0.1, 1] }}
                    transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
                  />

                  {/* Eyebrows */}
                  <div className="absolute top-9 left-3 w-2.5 h-0.5 bg-gray-700 rounded"></div>
                  <div className="absolute top-9 right-3 w-2.5 h-0.5 bg-gray-700 rounded"></div>

                  {/* Nose */}
                  <div className="absolute top-12 left-1/2 transform -translate-x-1/2 w-1 h-2 bg-orange-300 rounded"></div>

                  {/* Mouth */}
                  <motion.div
                    className="absolute top-14 left-1/2 transform -translate-x-1/2 w-3 h-1 bg-red-400 rounded-full"
                    animate={{ scaleX: [1, 1.2, 1] }}
                    transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
                  />

                  {/* Facial Hair/Stubble */}
                  <div className="absolute bottom-2 left-1/2 transform -translate-x-1/2 w-8 h-3 bg-gray-600 opacity-30 rounded-b-full"></div>
                </div>

                {/* Animated overlay for liveliness */}
                <motion.div
                  className="absolute inset-0 bg-gradient-to-t from-transparent via-transparent to-white/10"
                  animate={{
                    opacity: [0, 0.2, 0],
                  }}
                  transition={{
                    duration: 3,
                    repeat: Infinity,
                    ease: "easeInOut"
                  }}
                />
              </motion.div>

              {/* Enhanced Animated Arms */}
              <motion.div
                className="absolute -left-10 top-1/2 transform -translate-y-1/2 text-3xl"
                animate={{
                  rotate: [0, 25, -25, 0],
                  scale: [1, 1.1, 1]
                }}
                transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
              >
                👋
              </motion.div>

              <motion.div
                className="absolute -right-10 top-1/2 transform -translate-y-1/2 text-3xl"
                animate={{
                  rotate: [0, -25, 25, 0],
                  scale: [1, 1.1, 1]
                }}
                transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut", delay: 0.5 }}
              >
                👍
              </motion.div>

              {/* Enhanced sparkle effects */}
              <motion.div
                className="absolute -top-6 -right-6 text-yellow-300 text-3xl"
                animate={{
                  scale: [1, 1.5, 1],
                  rotate: [0, 360, 720],
                  y: [0, -15, 0]
                }}
                transition={{ duration: 3, repeat: Infinity }}
              >
                ⭐
              </motion.div>

              <motion.div
                className="absolute -bottom-4 -left-4 text-yellow-300 text-2xl"
                animate={{
                  scale: [1, 1.3, 1],
                  opacity: [1, 0.5, 1],
                  x: [0, 10, 0],
                  rotate: [0, 180, 360]
                }}
                transition={{ duration: 2.5, repeat: Infinity, delay: 0.7 }}
              >
                ✨
              </motion.div>

              <motion.div
                className="absolute top-2 left-2 text-yellow-300 text-xl"
                animate={{
                  scale: [1, 1.4, 1],
                  rotate: [0, 180, 360],
                  opacity: [1, 0.7, 1]
                }}
                transition={{ duration: 4, repeat: Infinity, delay: 1 }}
              >
                💫
              </motion.div>

              {/* Additional floating elements */}
              <motion.div
                className="absolute -top-8 left-1/2 transform -translate-x-1/2 text-2xl"
                animate={{
                  y: [0, -20, 0],
                  opacity: [0.7, 1, 0.7],
                  scale: [1, 1.2, 1]
                }}
                transition={{ duration: 3.5, repeat: Infinity, delay: 1.5 }}
              >
                🎉
              </motion.div>
            </motion.div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default Mascot;

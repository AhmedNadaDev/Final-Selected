import { useState, useEffect } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Clapperboard, Zap, FlaskConical, Menu, X } from 'lucide-react'

const links = [
  { to: '/',         label: 'Home',      icon: Clapperboard },
  { to: '/generate', label: 'Generate',  icon: Zap },
  { to: '/research', label: 'Research',  icon: FlaskConical },
]

export default function Navbar() {
  const { pathname } = useLocation()
  const [scrolled, setScrolled] = useState(false)
  const [open, setOpen] = useState(false)

  useEffect(() => {
    const fn = () => setScrolled(window.scrollY > 20)
    window.addEventListener('scroll', fn)
    return () => window.removeEventListener('scroll', fn)
  }, [])

  return (
    <motion.nav
      initial={{ y: -80, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
      className={`fixed top-0 inset-x-0 z-50 transition-all duration-500 ${
        scrolled ? 'glass border-b border-border' : 'bg-transparent'
      }`}
    >
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        {/* Logo */}
        <Link to="/" className="flex items-center gap-2.5 group">
          <div className="w-8 h-8 rounded-lg bg-plasma flex items-center justify-center
                          group-hover:shadow-[0_0_16px_rgba(124,58,237,0.7)] transition-shadow">
            <Clapperboard size={16} className="text-white" />
          </div>
          <span className="font-display font-bold text-lg tracking-tight">
            Nova<span className="gradient-text-plasma">Cine</span>
          </span>
        </Link>

        {/* Desktop links */}
        <div className="hidden md:flex items-center gap-1">
          {links.map(({ to, label, icon: Icon }) => {
            const active = pathname === to
            return (
              <Link
                key={to}
                to={to}
                className={`relative px-4 py-2 rounded-lg font-body text-sm font-medium
                            flex items-center gap-2 transition-colors duration-200
                            ${active ? 'text-neon' : 'text-subtle hover:text-text'}`}
              >
                <Icon size={14} />
                {label}
                {active && (
                  <motion.div
                    layoutId="nav-pill"
                    className="absolute inset-0 rounded-lg bg-plasma/10 border border-plasma/20"
                    style={{ zIndex: -1 }}
                    transition={{ type: 'spring', bounce: 0.2, duration: 0.4 }}
                  />
                )}
              </Link>
            )
          })}
        </div>

        {/* CTA */}
        <div className="hidden md:flex items-center gap-3">
          <Link
            to="/generate"
            className="btn-plasma px-5 py-2 rounded-lg text-sm text-white flex items-center gap-2"
          >
            <Zap size={14} />
            Start Generating
          </Link>
        </div>

        {/* Mobile hamburger */}
        <button className="md:hidden text-subtle hover:text-text" onClick={() => setOpen(!open)}>
          {open ? <X size={22} /> : <Menu size={22} />}
        </button>
      </div>

      {/* Mobile menu */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="md:hidden glass border-t border-border overflow-hidden"
          >
            <div className="px-6 py-4 flex flex-col gap-2">
              {links.map(({ to, label, icon: Icon }) => (
                <Link
                  key={to}
                  to={to}
                  onClick={() => setOpen(false)}
                  className="flex items-center gap-3 px-4 py-3 rounded-lg
                             text-text hover:bg-card transition-colors"
                >
                  <Icon size={16} className="text-neon" />
                  {label}
                </Link>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.nav>
  )
}

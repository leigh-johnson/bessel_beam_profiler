# Free-space laser diode → single-mode fiber alignment and coupling

Laser diode: CPS650F (elliptical beam shape), nominal wavelength = 650nm
Coupler: Kirchoff 60SMS
Fiber: Single mode fiber patch cable rated for 650nm. Do NOT use polarization-maintaining (PM) fiber.

1. Roughly collimate laser diode by twisting focus knob on laser diode.
2. Restore all mirrors to a neutral position by twisting adjustment knobs.
3. Coarsely align two reflective mirrors to form a 90 deg angles.
   1. Position mirror #1 at roughly a 45deg angle. Use a Thorlabs index card or ruler to rotate the mirror pedestal until beam is parallel to the table (check at multiple points). 
   2. Tightly screw down mirror, using T-hex
   3. Repeat for mirror #2.
   
4. Inspect fiber tips with handheld fiber microscope. If tips are dirty, clean by gently moving fiber tip in a figure-8 pattern across Thorlabs fiber optic cleaning cloth, which is kept on a blue spool (follow directions on the spool). The microscope and spool of fiber optic cleaning cloth are stored in the top shelf of the grey cabinet in the control room.

5. Connect fiber pen (Excimer Visual Fault Locator) to one end of fiber. Wear protective goggles during alignment procedure, since Fiber Pen is 10-24mW.
6. Turn on "glint" or "blink mode" on the fiber pen. 
7. Walk the two beams until they are aligned.
   1. Place index card in front of coupler (before diode laser enters coupler). Using mirror #1 (closest to laser), align the blinking beam and laser diode beam.
   2. Place the index cart in front of the diode laser beam face. Using mirror #2 (closest to coupler), align the blinking beam and laser diode beam.
   3. Repeat steps 1-2 until beam is aligned.
   
8. (If needed) Adjust the focus Kirchoff 60SMS fiber coupler. See "3.2 Changing the focus setting" in `datasheets/Adjustment_60SMS.pdf`. Note: locking the adjustment screws (see Figure 12) in the final step can result in signal loss, because the grub screws are not mechanically isolated.

Expect 1.2 - 2.0 mW after fiber is coupled to CPS650F diode laser.
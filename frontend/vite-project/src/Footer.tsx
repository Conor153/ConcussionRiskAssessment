function Footer() {
  // Funtion to display footer
  return (
    <div className="bg-primary w-full mt-auto items-center justify-between flex p-4">
      <img
        className="h-32 w-auto"
        src="src/assets/ATU-Logo-Full-RGB-White.svg"
      />
      <p className="text-white text-xl font-bold text-center">
        {" "}
        Conor Callaghan L00173495 2026 &copy;
      </p>
      <img
        className="bg-primary h-32 w-auto"
        src="src/assets/qr-code.svg"
      />
    </div>
  );
}

export default Footer;
